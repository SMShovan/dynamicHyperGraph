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
// Type-3 Motif:  For each triple (i,j,k) with i<j<k that form a triangle
// in H2H, compute x*y*z where:
//   A = con(i,j), B = con(j,k), C = con(i,k), G = group(i,j,k)
//   x = A - G, y = B - G, z = C - G
// Uses H2V and H2H.
// ---------------------------------------------------------------------------

static inline void checkCudaLocal(cudaError_t result) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result)
              << std::endl;
    std::exit(-1);
  }
}

// Baseline kernel: one thread per hyperedge, walks CBST nodes
__global__ void countType3Kernel(const CBSTNode *__restrict__ d_h2vNodes,
                                 const int *__restrict__ d_h2vFlat,
                                 const CBSTNode *__restrict__ d_h2hNodes,
                                 const int *__restrict__ d_h2hFlat,
                                 int numRecords, int fixedSize,
                                 long long *__restrict__ outCounts) {
  int i0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i0 >= numRecords)
    return;
  int i = i0 + 1; // 1-based key

  // CBST walk for H2H[i]
  const CBSTNode *h2h_node_i = d_h2hNodes;
  while (h2h_node_i != nullptr && h2h_node_i->index != i) {
    if (h2h_node_i->index > i)
      h2h_node_i = h2h_node_i->left;
    else
      h2h_node_i = h2h_node_i->right;
  }
  if (h2h_node_i == nullptr)
    return;
  int adjLoc_i = h2h_node_i->value;

  long long acc = 0;

  // Iterate neighbors j of i (j > i)
  for (int posJ = adjLoc_i; posJ < fixedSize; ++posJ) {
    int j_val = readFlat(d_h2hFlat, posJ);
    if (j_val == 0 || j_val == INT_MIN)
      break;
    if (j_val <= i)
      continue; // enforce i < j
    int j = j_val;

    // CBST walk for H2H[j]
    const CBSTNode *h2h_node_j = d_h2hNodes;
    while (h2h_node_j != nullptr && h2h_node_j->index != j) {
      if (h2h_node_j->index > j)
        h2h_node_j = h2h_node_j->left;
      else
        h2h_node_j = h2h_node_j->right;
    }
    if (h2h_node_j == nullptr)
      continue;
    int adjLoc_j = h2h_node_j->value;

    // Intersect adj(i) and adj(j) to find common k > j
    int p = adjLoc_i, q = adjLoc_j;
    while (p < fixedSize && q < fixedSize) {
      int vi = readFlat(d_h2hFlat, p);
      int vj = readFlat(d_h2hFlat, q);
      if (vi == 0 || vi == INT_MIN || vj == 0 || vj == INT_MIN)
        break;
      if (vi == vj) {
        int k = vi;
        if (k > j) {
          // Compute contributions for triple (i, j, k)
          // CBST walks for H2V
          const CBSTNode *node_a = d_h2vNodes;
          while (node_a != nullptr && node_a->index != i) {
            if (node_a->index > i)
              node_a = node_a->left;
            else
              node_a = node_a->right;
          }
          const CBSTNode *node_b = d_h2vNodes;
          while (node_b != nullptr && node_b->index != j) {
            if (node_b->index > j)
              node_b = node_b->left;
            else
              node_b = node_b->right;
          }
          const CBSTNode *node_c = d_h2vNodes;
          while (node_c != nullptr && node_c->index != k) {
            if (node_c->index > k)
              node_c = node_c->left;
            else
              node_c = node_c->right;
          }
          if (node_a && node_b && node_c) {
            int loc_a = node_a->value;
            int loc_b = node_b->value;
            int loc_c = node_c->value;
            int A = con((int *)d_h2vFlat, loc_a, loc_b);
            int B = con((int *)d_h2vFlat, loc_b, loc_c);
            int C = con((int *)d_h2vFlat, loc_a, loc_c);
            int G = group((int *)d_h2vFlat, loc_a, loc_b, loc_c);
            int x = A - G;
            int y = B - G;
            int z = C - G;
            if (x > 0 && y > 0 && z > 0) {
              acc += static_cast<long long>(x) * static_cast<long long>(y) *
                     static_cast<long long>(z);
            }
          }
        }
        ++p;
        ++q;
      } else if (vi < vj) {
        ++p;
      } else {
        ++q;
      }
    }
  }

  outCounts[i0] = acc;
}

// ---------------------------------------------------------------------------
// Delta kernels: subtract/add for affected hyperedges (frontier-based)
// ---------------------------------------------------------------------------
template <bool IsAddition>
__global__ void type3FrontierKernel(
    const CBSTNode *__restrict__ d_h2vNodes, const int *__restrict__ d_h2vFlat,
    const CBSTNode *__restrict__ d_h2hNodes, const int *__restrict__ d_h2hFlat,
    const int *__restrict__ d_frontierIds, // 1-based
    int frontierSize, int fixedSize, long long *__restrict__ d_delta) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= frontierSize)
    return;
  int i = d_frontierIds[tid];

  // CBST walk for H2H[i]
  const CBSTNode *h2h_node_i = d_h2hNodes;
  while (h2h_node_i != nullptr && h2h_node_i->index != i) {
    if (h2h_node_i->index > i)
      h2h_node_i = h2h_node_i->left;
    else
      h2h_node_i = h2h_node_i->right;
  }
  if (h2h_node_i == nullptr)
    return;
  int adjLoc_i = h2h_node_i->value;

  long long acc = 0;

  for (int posJ = adjLoc_i; posJ < fixedSize; ++posJ) {
    int j_val = readFlat(d_h2hFlat, posJ);
    if (j_val == 0 || j_val == INT_MIN)
      break;
    if (j_val <= i)
      continue;
    int j = j_val;

    const CBSTNode *h2h_node_j = d_h2hNodes;
    while (h2h_node_j != nullptr && h2h_node_j->index != j) {
      if (h2h_node_j->index > j)
        h2h_node_j = h2h_node_j->left;
      else
        h2h_node_j = h2h_node_j->right;
    }
    if (h2h_node_j == nullptr)
      continue;
    int adjLoc_j = h2h_node_j->value;

    int p = adjLoc_i, q = adjLoc_j;
    while (p < fixedSize && q < fixedSize) {
      int vi = readFlat(d_h2hFlat, p);
      int vj = readFlat(d_h2hFlat, q);
      if (vi == 0 || vi == INT_MIN || vj == 0 || vj == INT_MIN)
        break;
      if (vi == vj) {
        int k = vi;
        if (k > j) {
          const CBSTNode *node_a = d_h2vNodes;
          while (node_a != nullptr && node_a->index != i) {
            if (node_a->index > i)
              node_a = node_a->left;
            else
              node_a = node_a->right;
          }
          const CBSTNode *node_b = d_h2vNodes;
          while (node_b != nullptr && node_b->index != j) {
            if (node_b->index > j)
              node_b = node_b->left;
            else
              node_b = node_b->right;
          }
          const CBSTNode *node_c = d_h2vNodes;
          while (node_c != nullptr && node_c->index != k) {
            if (node_c->index > k)
              node_c = node_c->left;
            else
              node_c = node_c->right;
          }
          if (node_a && node_b && node_c) {
            int loc_a = node_a->value;
            int loc_b = node_b->value;
            int loc_c = node_c->value;
            int A = con((int *)d_h2vFlat, loc_a, loc_b);
            int B = con((int *)d_h2vFlat, loc_b, loc_c);
            int C = con((int *)d_h2vFlat, loc_a, loc_c);
            int G = group((int *)d_h2vFlat, loc_a, loc_b, loc_c);
            int x = A - G;
            int y = B - G;
            int z = C - G;
            if (x > 0 && y > 0 && z > 0) {
              acc += static_cast<long long>(x) * static_cast<long long>(y) *
                     static_cast<long long>(z);
            }
          }
        }
        ++p;
        ++q;
      } else if (vi < vj) {
        ++p;
      } else {
        ++q;
      }
    }
  }

  if (IsAddition) {
    atomicAdd(d_delta, acc);
  } else {
    atomicAdd(d_delta, -acc);
  }
}

// ---------------------------------------------------------------------------
// Type3 Delta Accumulator
// ---------------------------------------------------------------------------
struct Type3DeltaAccumulator {
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

static void type3Subtract(const CBSTContext &h2vCtx, const CBSTContext &h2hCtx,
                          const std::vector<int> &deletedIds,
                          Type3DeltaAccumulator &acc) {
  if (deletedIds.empty())
    return;
  int *d_frontier = nullptr;
  checkCudaLocal(cudaMalloc(&d_frontier, deletedIds.size() * sizeof(int)));
  checkCudaLocal(cudaMemcpy(d_frontier, deletedIds.data(),
                            deletedIds.size() * sizeof(int),
                            cudaMemcpyHostToDevice));
  int block = 256;
  int grid = (static_cast<int>(deletedIds.size()) + block - 1) / block;
  type3FrontierKernel<false><<<grid, block>>>(
      h2vCtx.d_nodes, h2vCtx.d_flatPayload, h2hCtx.d_nodes,
      h2hCtx.d_flatPayload, d_frontier, static_cast<int>(deletedIds.size()),
      h2hCtx.fixedSize, acc.d_delta);
  checkCudaLocal(cudaDeviceSynchronize());
  checkCudaLocal(cudaFree(d_frontier));
}

static void type3Add(const CBSTContext &h2vCtx, const CBSTContext &h2hCtx,
                     const std::vector<int> &insertedIds,
                     Type3DeltaAccumulator &acc) {
  if (insertedIds.empty())
    return;
  int *d_frontier = nullptr;
  checkCudaLocal(cudaMalloc(&d_frontier, insertedIds.size() * sizeof(int)));
  checkCudaLocal(cudaMemcpy(d_frontier, insertedIds.data(),
                            insertedIds.size() * sizeof(int),
                            cudaMemcpyHostToDevice));
  int block = 256;
  int grid = (static_cast<int>(insertedIds.size()) + block - 1) / block;
  type3FrontierKernel<true><<<grid, block>>>(
      h2vCtx.d_nodes, h2vCtx.d_flatPayload, h2hCtx.d_nodes,
      h2hCtx.d_flatPayload, d_frontier, static_cast<int>(insertedIds.size()),
      h2hCtx.fixedSize, acc.d_delta);
  checkCudaLocal(cudaDeviceSynchronize());
  checkCudaLocal(cudaFree(d_frontier));
}

// ---------------------------------------------------------------------------
// main: full pipeline matching main.cu pattern
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
  HypergraphParams params;
  if (!parseCommandLineArgs(argc, argv, params)) {
    return 1;
  }
  printHypergraphParams(params);

  auto [hyperedgeToVertex, vertexToHyperedge] = generateHypergraph(params);
  std::vector<std::vector<int>> hyperedge2hyperedge =
      hyperedgeAdjacency(vertexToHyperedge, hyperedgeToVertex);

  auto [h2vFlatVertexIds, h2vStartOffsets] =
      flatten(hyperedgeToVertex, "Hyperedge to Vertex");
  auto [v2hFlatHyperedgeIds, v2hStartOffsets] =
      flatten(vertexToHyperedge, "Vertex to Hyperedge");
  auto [h2hFlatAdjacency, h2hStartOffsets] =
      flatten(hyperedge2hyperedge, "Hyperedge to Hyperedge");

  auto [cbstH2VStartOffsets, cbstH2VKeys] = prepareCBSTData(h2vStartOffsets);
  auto [cbstV2HStartOffsets, cbstV2HKeys] = prepareCBSTData(v2hStartOffsets);
  auto [cbstH2HStartOffsets, cbstH2HKeys] = prepareCBSTData(h2hStartOffsets);

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
  // Baseline Type3 count
  // ---------------------
  long long *d_counts = nullptr;
  checkCudaLocal(
      cudaMalloc(&d_counts, params.numHyperedges * sizeof(long long)));
  checkCudaLocal(
      cudaMemset(d_counts, 0, params.numHyperedges * sizeof(long long)));
  int block = 256;
  int grid = (params.numHyperedges + block - 1) / block;
  countType3Kernel<<<grid, block>>>(
      h2vOps.context().d_nodes, h2vOps.context().d_flatPayload,
      h2hOps.context().d_nodes, h2hOps.context().d_flatPayload,
      params.numHyperedges, h2hOps.context().fixedSize, d_counts);
  checkCudaLocal(cudaDeviceSynchronize());
  std::vector<long long> counts(params.numHyperedges);
  checkCudaLocal(cudaMemcpy(counts.data(), d_counts,
                            params.numHyperedges * sizeof(long long),
                            cudaMemcpyDeviceToHost));
  long long totalBaseline = 0;
  for (long long c : counts)
    totalBaseline += c;
  std::cout << "[GPU] Type3 baseline motif count: " << totalBaseline
            << std::endl;
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
  Type3DeltaAccumulator type3Acc;
  type3Acc.init();
  type3Subtract(h2vOps.context(), h2hOps.context(), deletedIds, type3Acc);

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
  type3Add(h2vOpsNew.context(), h2hOpsNew.context(), mappedInsertIds, type3Acc);
  long long delta = type3Acc.read();
  std::cout << "[GPU] Type3 motif delta count: " << delta << std::endl;

  // Clean up memory
  delete[] cbstH2VKeys;
  delete[] cbstV2HKeys;
  delete[] cbstH2HKeys;

  return 0;
}
