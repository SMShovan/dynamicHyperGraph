#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cuda_runtime.h>
#include <iostream>
#include <set>
#include <unordered_map>
#include <vector>
// Include Thrust headers
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/scan.h>
// Include our utility functions
#include "../include/graphGeneration.hpp"
#include "../include/printUtils.hpp"
#include "../include/structure.hpp"
#include "../include/utils.hpp"

// Device helpers and moved kernels are provided via kernel headers
#include "../kernel/device_utils.cuh"
#include "../kernel/kernels.cuh"
#include "../kernel/motif_utils.cuh"

// ---------------------------------------------------------------------------
// Type-1 Motif:  For each hyperedge e, count C(|e|, 3) = |e|*(|e|-1)*(|e|-2)/6
// Uses H2V only.
// ---------------------------------------------------------------------------

static inline void checkCudaLocal(cudaError_t result) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result)
              << std::endl;
    std::exit(-1);
  }
}

// Baseline kernel: one thread per hyperedge, walks CBST to find payload start
__global__ void countType1Kernel(const CBSTNode *__restrict__ d_h2vNodes,
                                 const int *__restrict__ d_h2vFlat,
                                 int numRecords, int fixedSize,
                                 long long *__restrict__ outCounts) {
  int i0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i0 >= numRecords)
    return;
  int i = i0 + 1; // 1-based key

  // CBST walk to find payload start offset for hyperedge i
  const CBSTNode *node = d_h2vNodes;
  while (node != nullptr && node->index != i) {
    if (node->index > i)
      node = node->left;
    else
      node = node->right;
  }
  if (node == nullptr)
    return;
  int loc = node->value;

  // Compute degree using chain-pointer-aware deg()
  int n = deg((int *)d_h2vFlat, loc);

  long long c = 0;
  if (n >= 3) {
    long long nn = n;
    c = (nn * (nn - 1) * (nn - 2)) / 6;
  }
  outCounts[i0] = c;
}

// ---------------------------------------------------------------------------
// Delta kernels: subtract/add for affected hyperedges
// ---------------------------------------------------------------------------
template <bool IsAddition>
__global__ void
type1FrontierKernel(const CBSTNode *__restrict__ d_h2vNodes,
                    const int *__restrict__ d_h2vFlat,
                    const int *__restrict__ d_frontierIds, // 1-based
                    int frontierSize, long long *__restrict__ d_delta) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= frontierSize)
    return;
  int i = d_frontierIds[tid];

  const CBSTNode *node = d_h2vNodes;
  while (node != nullptr && node->index != i) {
    if (node->index > i)
      node = node->left;
    else
      node = node->right;
  }
  if (node == nullptr)
    return;
  int loc = node->value;

  int n = deg((int *)d_h2vFlat, loc);
  long long c = 0;
  if (n >= 3) {
    long long nn = n;
    c = (nn * (nn - 1) * (nn - 2)) / 6;
  }
  if (IsAddition) {
    atomicAdd(d_delta, c);
  } else {
    atomicAdd(d_delta, -c);
  }
}

// ---------------------------------------------------------------------------
// Type1 Delta Accumulator
// ---------------------------------------------------------------------------
struct Type1DeltaAccumulator {
  long long *d_delta = nullptr;
  void init() {
    checkCudaLocal(cudaMalloc(&d_delta, sizeof(long long)));
    checkCudaLocal(cudaMemset(d_delta, 0, sizeof(long long)));
  }
  long long read() {
    long long val = 0;
    checkCudaLocal(
        cudaMemcpy(&val, d_delta, sizeof(long long), cudaMemcpyDeviceToHost));
    checkCudaLocal(cudaFree(d_delta));
    d_delta = nullptr;
    return val;
  }
};

static void type1Subtract(const CBSTContext &h2vCtx,
                          const std::vector<int> &deletedIds,
                          Type1DeltaAccumulator &acc) {
  if (deletedIds.empty())
    return;
  int *d_frontier = nullptr;
  checkCudaLocal(cudaMalloc(&d_frontier, deletedIds.size() * sizeof(int)));
  checkCudaLocal(cudaMemcpy(d_frontier, deletedIds.data(),
                            deletedIds.size() * sizeof(int),
                            cudaMemcpyHostToDevice));
  int block = 256;
  int grid = (static_cast<int>(deletedIds.size()) + block - 1) / block;
  type1FrontierKernel<false>
      <<<grid, block>>>(h2vCtx.d_nodes, h2vCtx.d_flatPayload, d_frontier,
                        static_cast<int>(deletedIds.size()), acc.d_delta);
  checkCudaLocal(cudaDeviceSynchronize());
  checkCudaLocal(cudaFree(d_frontier));
}

static void type1Add(const CBSTContext &h2vCtx,
                     const std::vector<int> &insertedIds,
                     Type1DeltaAccumulator &acc) {
  if (insertedIds.empty())
    return;
  int *d_frontier = nullptr;
  checkCudaLocal(cudaMalloc(&d_frontier, insertedIds.size() * sizeof(int)));
  checkCudaLocal(cudaMemcpy(d_frontier, insertedIds.data(),
                            insertedIds.size() * sizeof(int),
                            cudaMemcpyHostToDevice));
  int block = 256;
  int grid = (static_cast<int>(insertedIds.size()) + block - 1) / block;
  type1FrontierKernel<true>
      <<<grid, block>>>(h2vCtx.d_nodes, h2vCtx.d_flatPayload, d_frontier,
                        static_cast<int>(insertedIds.size()), acc.d_delta);
  checkCudaLocal(cudaDeviceSynchronize());
  checkCudaLocal(cudaFree(d_frontier));
}

// ---------------------------------------------------------------------------
// main: full pipeline matching main.cu pattern
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
  // Parse command line arguments
  HypergraphParams params;
  if (!parseCommandLineArgs(argc, argv, params)) {
    return 1;
  }
  printHypergraphParams(params);

  // Generate hypergraph mappings
  auto [hyperedgeToVertex, vertexToHyperedge] = generateHypergraph(params);

  // Generate hyperedge-to-hyperedge adjacency
  std::vector<std::vector<int>> hyperedge2hyperedge =
      hyperedgeAdjacency(vertexToHyperedge, hyperedgeToVertex);

  // Flatten the 2D vectors for GPU processing
  auto [h2vFlatVertexIds, h2vStartOffsets] =
      flatten(hyperedgeToVertex, "Hyperedge to Vertex");
  auto [v2hFlatHyperedgeIds, v2hStartOffsets] =
      flatten(vertexToHyperedge, "Vertex to Hyperedge");
  auto [h2hFlatAdjacency, h2hStartOffsets] =
      flatten(hyperedge2hyperedge, "Hyperedge to Hyperedge");

  // Prepare data for Complete Binary Search Tree construction
  auto [cbstH2VStartOffsets, cbstH2VKeys] = prepareCBSTData(h2vStartOffsets);
  auto [cbstV2HStartOffsets, cbstV2HKeys] = prepareCBSTData(v2hStartOffsets);
  auto [cbstH2HStartOffsets, cbstH2HKeys] = prepareCBSTData(h2hStartOffsets);

  // Construct Complete Binary Search Trees
  int numVertices = static_cast<int>(vertexToHyperedge.size());

  CBSTOperations h2vOps("H2V", params.payloadCapacity, params.alignment);
  h2vOps.construct(cbstH2VKeys, cbstH2VStartOffsets, params.numHyperedges,
                   h2vFlatVertexIds.data(),
                   static_cast<int>(h2vFlatVertexIds.size()));

  CBSTOperations v2hOps("V2H", params.payloadCapacity, params.alignment);
  v2hOps.construct(cbstV2HKeys, cbstV2HStartOffsets, numVertices,
                   v2hFlatHyperedgeIds.data(),
                   static_cast<int>(v2hFlatHyperedgeIds.size()));

  CBSTOperations h2hOps("H2H", params.payloadCapacity, params.alignment);
  h2hOps.construct(cbstH2HKeys, cbstH2HStartOffsets, params.numHyperedges,
                   h2hFlatAdjacency.data(),
                   static_cast<int>(h2hFlatAdjacency.size()));

  // ---------------------
  // Baseline Type1 count
  // ---------------------
  long long *d_counts = nullptr;
  checkCudaLocal(
      cudaMalloc(&d_counts, params.numHyperedges * sizeof(long long)));
  checkCudaLocal(
      cudaMemset(d_counts, 0, params.numHyperedges * sizeof(long long)));
  int block = 256;
  int grid = (params.numHyperedges + block - 1) / block;
  countType1Kernel<<<grid, block>>>(
      h2vOps.context().d_nodes, h2vOps.context().d_flatPayload,
      params.numHyperedges, h2vOps.context().fixedSize, d_counts);
  checkCudaLocal(cudaDeviceSynchronize());
  std::vector<long long> counts(params.numHyperedges);
  checkCudaLocal(cudaMemcpy(counts.data(), d_counts,
                            params.numHyperedges * sizeof(long long),
                            cudaMemcpyDeviceToHost));
  long long totalBaseline = 0;
  for (long long c : counts)
    totalBaseline += c;
  std::cout << "[GPU] Type1 baseline motif count (sum_e C(|e|, 3)): "
            << totalBaseline << std::endl;
  checkCudaLocal(cudaFree(d_counts));

  // --------------------------
  // DeltaGeneration()
  // --------------------------
  int N = params.numHyperedges;
  int numDeletes = std::max(1, std::min(2, N));
  std::vector<int> deletedIds;
  for (int k = 0; k < numDeletes; ++k)
    deletedIds.push_back(N - k);
  std::sort(deletedIds.begin(), deletedIds.end());

  int numInserts = numDeletes + 1;
  auto generatedInserts =
      hyperedge2vertex(numInserts, params.maxVerticesPerHyperedge,
                       params.minVertexId, params.maxVertexId);
  int reuseK = std::min(numDeletes, numInserts);
  std::vector<int> insertAssignedIds(numInserts);
  for (int i = 0; i < reuseK; ++i)
    insertAssignedIds[i] = deletedIds[i];
  for (int i = reuseK; i < numInserts; ++i)
    insertAssignedIds[i] = N + (i - reuseK) + 1;

  // --------------------------
  // DataStructureUpdate() (host model)
  // --------------------------
  std::unordered_map<int, std::vector<int>> vRem;
  for (int hId : deletedIds) {
    if (hId >= 1 && hId <= static_cast<int>(hyperedgeToVertex.size())) {
      for (int v : hyperedgeToVertex[hId - 1])
        vRem[v].push_back(hId);
    }
  }
  std::vector<int> v2hRemoveKeys, v2hRemoveValues, v2hRemovePrefix;
  v2hRemoveKeys.reserve(vRem.size());
  for (auto &kv : vRem) {
    v2hRemoveKeys.push_back(kv.first);
    for (int h : kv.second)
      v2hRemoveValues.push_back(h);
    int newSize = (v2hRemovePrefix.empty() ? 0 : v2hRemovePrefix.back()) +
                  static_cast<int>(kv.second.size());
    v2hRemovePrefix.push_back(newSize);
  }

  std::unordered_map<int, std::vector<int>> vIns;
  for (size_t i = 0; i < generatedInserts.size(); ++i) {
    int hId = insertAssignedIds[i];
    for (int v : generatedInserts[i])
      vIns[v].push_back(hId);
  }
  std::vector<int> v2hInsertKeys, v2hInsertValues, v2hInsertPrefix;
  v2hInsertKeys.reserve(vIns.size());
  for (auto &kv : vIns) {
    v2hInsertKeys.push_back(kv.first);
    for (int h : kv.second)
      v2hInsertValues.push_back(h);
    int newSize = (v2hInsertPrefix.empty() ? 0 : v2hInsertPrefix.back()) +
                  static_cast<int>(kv.second.size());
    v2hInsertPrefix.push_back(newSize);
  }

  // --------------------------
  // CountUpdate() -- Phase A: subtract BEFORE modifying data structures
  // --------------------------
  Type1DeltaAccumulator type1Acc;
  type1Acc.init();
  type1Subtract(h2vOps.context(), deletedIds, type1Acc);

  // --------------------------
  // DataStructureUpdate() -- modify CBST data structures
  // --------------------------
  h2vOps.erase(deletedIds);
  if (!v2hRemoveKeys.empty()) {
    unfillCBST(v2hRemoveKeys, v2hRemoveValues, v2hRemovePrefix,
               const_cast<CBSTContext &>(v2hOps.context()));
  }

  std::vector<int> h2vInsertKeys = insertAssignedIds;
  std::vector<int> h2vInsertPayload;
  std::vector<int> h2vInsertPrefix;
  for (size_t i = 0; i < generatedInserts.size(); ++i) {
    for (int v : generatedInserts[i])
      h2vInsertPayload.push_back(v);
    int newSize = (h2vInsertPrefix.empty() ? 0 : h2vInsertPrefix.back()) +
                  static_cast<int>(generatedInserts[i].size());
    h2vInsertPrefix.push_back(newSize);
  }
  InsertMapping h2vMapping =
      h2vOps.insert(h2vInsertKeys, h2vInsertPayload, h2vInsertPrefix);

  if (!v2hInsertKeys.empty()) {
    fillCBST(v2hInsertKeys, v2hInsertValues, v2hInsertPrefix,
             const_cast<CBSTContext &>(v2hOps.context()));
  }

  // Host updated structures for rebuild
  int maxMappedKey = 0;
  for (int k : h2vMapping.itemToKey)
    maxMappedKey = std::max(maxMappedKey, k);
  int maxId = std::max(N, maxMappedKey);
  std::vector<std::vector<int>> updatedH2V = hyperedgeToVertex;
  if (static_cast<int>(updatedH2V.size()) < maxId)
    updatedH2V.resize(maxId);
  for (int hId : deletedIds) {
    if (hId >= 1 && hId <= static_cast<int>(updatedH2V.size()))
      updatedH2V[hId - 1].clear();
  }
  for (size_t i = 0; i < generatedInserts.size(); ++i) {
    int hId = h2vMapping.itemToKey[i];
    if (hId >= 1) {
      if (hId > static_cast<int>(updatedH2V.size()))
        updatedH2V.resize(hId);
      updatedH2V[hId - 1] = generatedInserts[i];
    }
  }
  auto updatedV2H = vertex2hyperedge(updatedH2V);
  auto updatedH2H = hyperedgeAdjacency(updatedV2H, updatedH2V);

  // Flatten and reconstruct updated CBSTs
  auto [h2vFlatValsNew, h2vStartsNew] =
      flatten(updatedH2V, "Updated Hyperedge to Vertex");
  auto [v2hFlatValsNew, v2hStartsNew] =
      flatten(updatedV2H, "Updated Vertex to Hyperedge");
  auto [h2hFlatValsNew, h2hStartsNew] =
      flatten(updatedH2H, "Updated Hyperedge to Hyperedge");
  auto [cbstH2VStartsNew, cbstH2VKeysNew] = prepareCBSTData(h2vStartsNew);
  auto [cbstV2HStartsNew, cbstV2HKeysNew] = prepareCBSTData(v2hStartsNew);
  auto [cbstH2HStartsNew, cbstH2HKeysNew] = prepareCBSTData(h2hStartsNew);

  CBSTOperations h2vOpsNew("H2V-new", params.payloadCapacity, params.alignment);
  h2vOpsNew.construct(cbstH2VKeysNew, cbstH2VStartsNew, maxId,
                      h2vFlatValsNew.data(),
                      static_cast<int>(h2vFlatValsNew.size()));
  CBSTOperations v2hOpsNew("V2H-new", params.payloadCapacity, params.alignment);
  v2hOpsNew.construct(
      cbstV2HKeysNew, cbstV2HStartsNew, static_cast<int>(updatedV2H.size()),
      v2hFlatValsNew.data(), static_cast<int>(v2hFlatValsNew.size()));
  CBSTOperations h2hOpsNew("H2H-new", params.payloadCapacity, params.alignment);
  h2hOpsNew.construct(cbstH2HKeysNew, cbstH2HStartsNew, maxId,
                      h2hFlatValsNew.data(),
                      static_cast<int>(h2hFlatValsNew.size()));

  // --------------------------
  // CountUpdate() -- Phase B: add using new snapshots
  // --------------------------
  std::vector<int> mappedInsertIds(h2vMapping.itemToKey.begin(),
                                   h2vMapping.itemToKey.end());
  type1Add(h2vOpsNew.context(), mappedInsertIds, type1Acc);
  long long delta = type1Acc.read();
  std::cout << "[GPU] Type1 motif delta count: " << delta << std::endl;

  // Clean up memory
  delete[] cbstH2VKeys;
  delete[] cbstV2HKeys;
  delete[] cbstH2HKeys;

  return 0;
}
