#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <set>
#include <vector>

#include "../include/graphGeneration.hpp"
#include "../include/structure.hpp"
#include "../include/utils.hpp"
#include "../kernel/device_utils.cuh"
#include "../kernel/motif_utils.cuh"

struct CoarseTriangleCounts {
  long long inner = 0;
  long long outer = 0;
  long long hybrid = 0;
  long long hyperedge = 0;
};

struct ScalarDeltaAccumulator {
  long long *d_value = nullptr;

  void init() {
    if (d_value != nullptr) {
      return;
    }
    cudaError_t alloc = cudaMalloc(&d_value, sizeof(long long));
    if (alloc != cudaSuccess) {
      std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(alloc)
                << std::endl;
      std::exit(-1);
    }
    cudaError_t zero = cudaMemset(d_value, 0, sizeof(long long));
    if (zero != cudaSuccess) {
      std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(zero)
                << std::endl;
      std::exit(-1);
    }
  }

  long long read() {
    long long value = 0;
    cudaError_t copy =
        cudaMemcpy(&value, d_value, sizeof(long long), cudaMemcpyDeviceToHost);
    if (copy != cudaSuccess) {
      std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(copy)
                << std::endl;
      std::exit(-1);
    }
    cudaError_t freeResult = cudaFree(d_value);
    if (freeResult != cudaSuccess) {
      std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(freeResult)
                << std::endl;
      std::exit(-1);
    }
    d_value = nullptr;
    return value;
  }
};

struct VertexSnapshots {
  std::vector<std::vector<int>> rawV2H;
  std::vector<std::vector<int>> rawV2V;
  std::vector<std::vector<int>> trimmedV2H;
  std::vector<std::vector<int>> trimmedV2V;
};

static inline void checkCudaLocal(cudaError_t result) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result)
              << std::endl;
    std::exit(-1);
  }
}

static std::vector<std::vector<int>> buildVertexPairwiseGraph(
    const std::vector<std::vector<int>> &hyperedgeToVertex) {
  int maxVertexId = 0;
  for (const auto &hyperedge : hyperedgeToVertex) {
    if (!hyperedge.empty()) {
      maxVertexId = std::max(maxVertexId, hyperedge.back());
    }
  }

  std::vector<std::set<int>> adjacency(maxVertexId + 1);
  for (const auto &hyperedge : hyperedgeToVertex) {
    for (size_t i = 0; i < hyperedge.size(); ++i) {
      for (size_t j = i + 1; j < hyperedge.size(); ++j) {
        int u = hyperedge[i];
        int v = hyperedge[j];
        adjacency[u].insert(v);
        adjacency[v].insert(u);
      }
    }
  }

  std::vector<std::vector<int>> vertexToVertex(maxVertexId + 1);
  for (int v = 1; v <= maxVertexId; ++v) {
    vertexToVertex[v] =
        std::vector<int>(adjacency[v].begin(), adjacency[v].end());
  }
  return vertexToVertex;
}

static std::vector<std::vector<int>>
trimZeroRow(const std::vector<std::vector<int>> &rows) {
  if (rows.empty()) {
    return {};
  }
  return std::vector<std::vector<int>>(rows.begin() + 1, rows.end());
}

static VertexSnapshots
buildVertexSnapshots(const std::vector<std::vector<int>> &hyperedgeToVertex) {
  VertexSnapshots snapshots;
  snapshots.rawV2H = vertex2hyperedge(hyperedgeToVertex);
  snapshots.rawV2V = buildVertexPairwiseGraph(hyperedgeToVertex);
  snapshots.trimmedV2H = trimZeroRow(snapshots.rawV2H);
  snapshots.trimmedV2V = trimZeroRow(snapshots.rawV2V);
  return snapshots;
}

static std::vector<int> collectVerticesFromHyperedges(
    const std::vector<std::vector<int>> &hyperedgeToVertex,
    const std::vector<int> &hyperedgeIds) {
  std::set<int> affected;
  for (int hyperedgeId : hyperedgeIds) {
    if (hyperedgeId < 1 ||
        hyperedgeId > static_cast<int>(hyperedgeToVertex.size())) {
      continue;
    }
    for (int vertexId : hyperedgeToVertex[hyperedgeId - 1]) {
      affected.insert(vertexId);
    }
  }
  return std::vector<int>(affected.begin(), affected.end());
}

static std::vector<int>
collectVerticesFromRows(const std::vector<std::vector<int>> &rows) {
  std::set<int> affected;
  for (const auto &row : rows) {
    for (int vertexId : row) {
      affected.insert(vertexId);
    }
  }
  return std::vector<int>(affected.begin(), affected.end());
}

static inline __device__ const CBSTNode *findNodeByKey(const CBSTNode *nodes,
                                                       int key) {
  const CBSTNode *current = nodes;
  while (current != nullptr && current->index != key) {
    current = (current->index > key) ? current->left : current->right;
  }
  return current;
}

static inline __device__ long long comb3Device(int n) {
  if (n < 3) {
    return 0;
  }
  long long nn = n;
  return (nn * (nn - 1) * (nn - 2)) / 6;
}

__global__ void
countInnerTrianglesKernel(const CBSTNode *__restrict__ d_h2vNodes,
                          const int *__restrict__ d_h2vFlat, int numRecords,
                          long long *__restrict__ outCounts) {
  int i0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i0 >= numRecords) {
    return;
  }
  int i = i0 + 1;
  const CBSTNode *node = findNodeByKey(d_h2vNodes, i);
  if (node == nullptr) {
    return;
  }
  int loc = node->value;
  int degree = deg((int *)d_h2vFlat, loc);
  outCounts[i0] = comb3Device(degree);
}

__global__ void countHyperedgeTrianglesKernel(
    const CBSTNode *__restrict__ d_h2hNodes, const int *__restrict__ d_h2hFlat,
    int numRecords, int fixedSize, long long *__restrict__ outCounts) {
  int i0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (i0 >= numRecords) {
    return;
  }
  int i = i0 + 1;
  const CBSTNode *node_i = findNodeByKey(d_h2hNodes, i);
  if (node_i == nullptr) {
    return;
  }
  int loc_i = node_i->value;
  long long acc = 0;

  for (int off_j = loc_i; off_j < fixedSize; ++off_j) {
    int j = readFlat(d_h2hFlat, off_j);
    if (j == 0 || j == INT_MIN) {
      break;
    }
    if (j <= i) {
      continue;
    }

    const CBSTNode *node_j = findNodeByKey(d_h2hNodes, j);
    if (node_j == nullptr) {
      continue;
    }
    int loc_j = node_j->value;

    int p = loc_i;
    int q = loc_j;
    while (true) {
      int a_val = readFlat(d_h2hFlat, p);
      int b_val = readFlat(d_h2hFlat, q);
      if (a_val == 0 || a_val == INT_MIN || b_val == 0 || b_val == INT_MIN) {
        break;
      }
      if (a_val == b_val) {
        int k = a_val;
        if (k > j) {
          ++acc;
        }
        ++p;
        ++q;
      } else if (a_val < b_val) {
        ++p;
      } else {
        ++q;
      }
      if (p >= fixedSize || q >= fixedSize) {
        break;
      }
    }
  }

  outCounts[i0] = acc;
}

__global__ void countOuterTrianglesKernel(
    const CBSTNode *__restrict__ d_v2vNodes, const int *__restrict__ d_v2vFlat,
    const CBSTNode *__restrict__ d_v2hNodes, const int *__restrict__ d_v2hFlat,
    int numRecords, int fixedSize, long long *__restrict__ outCounts) {
  int u0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (u0 >= numRecords) {
    return;
  }
  int u = u0 + 1;
  const CBSTNode *node_u = findNodeByKey(d_v2vNodes, u);
  if (node_u == nullptr) {
    return;
  }
  int loc_u = node_u->value;
  long long acc = 0;

  for (int off_v = loc_u; off_v < fixedSize; ++off_v) {
    int v = readFlat(d_v2vFlat, off_v);
    if (v == 0 || v == INT_MIN) {
      break;
    }
    if (v <= u) {
      continue;
    }

    const CBSTNode *node_v = findNodeByKey(d_v2vNodes, v);
    if (node_v == nullptr) {
      continue;
    }
    int loc_v = node_v->value;

    int p = loc_u;
    int q = loc_v;
    while (true) {
      int a_val = readFlat(d_v2vFlat, p);
      int b_val = readFlat(d_v2vFlat, q);
      if (a_val == 0 || a_val == INT_MIN || b_val == 0 || b_val == INT_MIN) {
        break;
      }
      if (a_val == b_val) {
        int w = a_val;
        if (w > v) {
          const CBSTNode *node_a = findNodeByKey(d_v2hNodes, u);
          const CBSTNode *node_b = findNodeByKey(d_v2hNodes, v);
          const CBSTNode *node_c = findNodeByKey(d_v2hNodes, w);
          if (node_a != nullptr && node_b != nullptr && node_c != nullptr) {
            int loc_a = node_a->value;
            int loc_b = node_b->value;
            int loc_c = node_c->value;
            if (group((int *)d_v2hFlat, loc_a, loc_b, loc_c) == 0) {
              ++acc;
            }
          }
        }
        ++p;
        ++q;
      } else if (a_val < b_val) {
        ++p;
      } else {
        ++q;
      }
      if (p >= fixedSize || q >= fixedSize) {
        break;
      }
    }
  }

  outCounts[u0] = acc;
}

__global__ void countHybridTrianglesKernel(
    const CBSTNode *__restrict__ d_v2vNodes, const int *__restrict__ d_v2vFlat,
    const CBSTNode *__restrict__ d_v2hNodes, const int *__restrict__ d_v2hFlat,
    int numRecords, int fixedSize, long long *__restrict__ outCounts) {
  int u0 = blockIdx.x * blockDim.x + threadIdx.x;
  if (u0 >= numRecords) {
    return;
  }
  int u = u0 + 1;
  const CBSTNode *node_u = findNodeByKey(d_v2vNodes, u);
  if (node_u == nullptr) {
    return;
  }
  int loc_u = node_u->value;
  long long acc = 0;

  for (int off_v = loc_u; off_v < fixedSize; ++off_v) {
    int v = readFlat(d_v2vFlat, off_v);
    if (v == 0 || v == INT_MIN) {
      break;
    }
    if (v <= u) {
      continue;
    }

    const CBSTNode *node_v = findNodeByKey(d_v2vNodes, v);
    if (node_v == nullptr) {
      continue;
    }
    int loc_v = node_v->value;

    int p = loc_u;
    int q = loc_v;
    while (true) {
      int a_val = readFlat(d_v2vFlat, p);
      int b_val = readFlat(d_v2vFlat, q);
      if (a_val == 0 || a_val == INT_MIN || b_val == 0 || b_val == INT_MIN) {
        break;
      }
      if (a_val == b_val) {
        int w = a_val;
        if (w > v) {
          const CBSTNode *node_a = findNodeByKey(d_v2hNodes, u);
          const CBSTNode *node_b = findNodeByKey(d_v2hNodes, v);
          const CBSTNode *node_c = findNodeByKey(d_v2hNodes, w);
          if (node_a != nullptr && node_b != nullptr && node_c != nullptr) {
            int loc_a = node_a->value;
            int loc_b = node_b->value;
            int loc_c = node_c->value;
            if (group((int *)d_v2hFlat, loc_a, loc_b, loc_c) > 0) {
              ++acc;
            }
          }
        }
        ++p;
        ++q;
      } else if (a_val < b_val) {
        ++p;
      } else {
        ++q;
      }
      if (p >= fixedSize || q >= fixedSize) {
        break;
      }
    }
  }

  outCounts[u0] = acc;
}

template <bool IsAddition>
__global__ void innerFrontierKernel(const CBSTNode *__restrict__ d_h2vNodes,
                                    const int *__restrict__ d_h2vFlat,
                                    const int *__restrict__ d_frontierIds,
                                    int frontierSize,
                                    long long *__restrict__ d_delta) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= frontierSize) {
    return;
  }
  int i = d_frontierIds[tid];
  const CBSTNode *node = findNodeByKey(d_h2vNodes, i);
  if (node == nullptr) {
    return;
  }
  int loc = node->value;
  long long delta = comb3Device(deg((int *)d_h2vFlat, loc));
  atomicAdd_sll(d_delta, IsAddition ? delta : -delta);
}

__global__ void buildHyperedgeAnchorFlags(
    const CBSTNode *__restrict__ d_h2hNodes, const int *__restrict__ d_h2hFlat,
    const int *__restrict__ d_frontierIds, int frontierSize, int fixedSize,
    int *__restrict__ d_anchorFlags) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= frontierSize) {
    return;
  }
  int f = d_frontierIds[tid];
  const CBSTNode *node_f = findNodeByKey(d_h2hNodes, f);
  if (node_f == nullptr) {
    return;
  }
  int loc_f = node_f->value;

  for (int off_j = loc_f; off_j < fixedSize; ++off_j) {
    int j = readFlat(d_h2hFlat, off_j);
    if (j == 0 || j == INT_MIN) {
      break;
    }

    const CBSTNode *node_j = findNodeByKey(d_h2hNodes, j);
    if (node_j == nullptr) {
      continue;
    }
    int loc_j = node_j->value;

    int p = loc_f;
    int q = loc_j;
    while (true) {
      int a_val = readFlat(d_h2hFlat, p);
      int b_val = readFlat(d_h2hFlat, q);
      if (a_val == 0 || a_val == INT_MIN || b_val == 0 || b_val == INT_MIN) {
        break;
      }
      if (a_val == b_val) {
        int k = a_val;
        int anchor = f;
        if (j < anchor) {
          anchor = j;
        }
        if (k < anchor) {
          anchor = k;
        }
        atomicExch(&d_anchorFlags[anchor], 1);
        ++p;
        ++q;
      } else if (a_val < b_val) {
        ++p;
      } else {
        ++q;
      }
      if (p >= fixedSize || q >= fixedSize) {
        break;
      }
    }
  }
}

template <bool IsAddition>
__global__ void
countHyperedgeAnchorsKernel(const CBSTNode *__restrict__ d_h2hNodes,
                            const int *__restrict__ d_h2hFlat,
                            const int *__restrict__ d_anchorIds, int anchorSize,
                            int fixedSize, long long *__restrict__ d_delta) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= anchorSize) {
    return;
  }
  int i = d_anchorIds[tid];
  const CBSTNode *node_i = findNodeByKey(d_h2hNodes, i);
  if (node_i == nullptr) {
    return;
  }
  int loc_i = node_i->value;
  long long acc = 0;

  for (int off_j = loc_i; off_j < fixedSize; ++off_j) {
    int j = readFlat(d_h2hFlat, off_j);
    if (j == 0 || j == INT_MIN) {
      break;
    }
    if (j <= i) {
      continue;
    }

    const CBSTNode *node_j = findNodeByKey(d_h2hNodes, j);
    if (node_j == nullptr) {
      continue;
    }
    int loc_j = node_j->value;

    int p = loc_i;
    int q = loc_j;
    while (true) {
      int a_val = readFlat(d_h2hFlat, p);
      int b_val = readFlat(d_h2hFlat, q);
      if (a_val == 0 || a_val == INT_MIN || b_val == 0 || b_val == INT_MIN) {
        break;
      }
      if (a_val == b_val) {
        int k = a_val;
        if (k > j) {
          ++acc;
        }
        ++p;
        ++q;
      } else if (a_val < b_val) {
        ++p;
      } else {
        ++q;
      }
      if (p >= fixedSize || q >= fixedSize) {
        break;
      }
    }
  }

  atomicAdd_sll(d_delta, IsAddition ? acc : -acc);
}

__global__ void buildOuterAnchorFlags(const CBSTNode *__restrict__ d_v2vNodes,
                                      const int *__restrict__ d_v2vFlat,
                                      const int *__restrict__ d_frontierIds,
                                      int frontierSize, int fixedSize,
                                      int *__restrict__ d_anchorFlags) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= frontierSize) {
    return;
  }
  int f = d_frontierIds[tid];
  const CBSTNode *node_f = findNodeByKey(d_v2vNodes, f);
  if (node_f == nullptr) {
    return;
  }
  int loc_f = node_f->value;

  for (int off_v = loc_f; off_v < fixedSize; ++off_v) {
    int v = readFlat(d_v2vFlat, off_v);
    if (v == 0 || v == INT_MIN) {
      break;
    }

    const CBSTNode *node_v = findNodeByKey(d_v2vNodes, v);
    if (node_v == nullptr) {
      continue;
    }
    int loc_v = node_v->value;

    int p = loc_f;
    int q = loc_v;
    while (true) {
      int a_val = readFlat(d_v2vFlat, p);
      int b_val = readFlat(d_v2vFlat, q);
      if (a_val == 0 || a_val == INT_MIN || b_val == 0 || b_val == INT_MIN) {
        break;
      }
      if (a_val == b_val) {
        int w = a_val;
        int anchor = f;
        if (v < anchor) {
          anchor = v;
        }
        if (w < anchor) {
          anchor = w;
        }
        atomicExch(&d_anchorFlags[anchor], 1);
        ++p;
        ++q;
      } else if (a_val < b_val) {
        ++p;
      } else {
        ++q;
      }
      if (p >= fixedSize || q >= fixedSize) {
        break;
      }
    }
  }
}

template <bool IsAddition>
__global__ void countOuterAnchorsKernel(const CBSTNode *__restrict__ d_v2vNodes,
                                        const int *__restrict__ d_v2vFlat,
                                        const CBSTNode *__restrict__ d_v2hNodes,
                                        const int *__restrict__ d_v2hFlat,
                                        const int *__restrict__ d_anchorIds,
                                        int anchorSize, int fixedSize,
                                        long long *__restrict__ d_delta) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= anchorSize) {
    return;
  }
  int u = d_anchorIds[tid];
  const CBSTNode *node_u = findNodeByKey(d_v2vNodes, u);
  if (node_u == nullptr) {
    return;
  }
  int loc_u = node_u->value;
  long long acc = 0;

  for (int off_v = loc_u; off_v < fixedSize; ++off_v) {
    int v = readFlat(d_v2vFlat, off_v);
    if (v == 0 || v == INT_MIN) {
      break;
    }
    if (v <= u) {
      continue;
    }

    const CBSTNode *node_v = findNodeByKey(d_v2vNodes, v);
    if (node_v == nullptr) {
      continue;
    }
    int loc_v = node_v->value;

    int p = loc_u;
    int q = loc_v;
    while (true) {
      int a_val = readFlat(d_v2vFlat, p);
      int b_val = readFlat(d_v2vFlat, q);
      if (a_val == 0 || a_val == INT_MIN || b_val == 0 || b_val == INT_MIN) {
        break;
      }
      if (a_val == b_val) {
        int w = a_val;
        if (w > v) {
          const CBSTNode *node_a = findNodeByKey(d_v2hNodes, u);
          const CBSTNode *node_b = findNodeByKey(d_v2hNodes, v);
          const CBSTNode *node_c = findNodeByKey(d_v2hNodes, w);
          if (node_a != nullptr && node_b != nullptr && node_c != nullptr) {
            int loc_a = node_a->value;
            int loc_b = node_b->value;
            int loc_c = node_c->value;
            if (group((int *)d_v2hFlat, loc_a, loc_b, loc_c) == 0) {
              ++acc;
            }
          }
        }
        ++p;
        ++q;
      } else if (a_val < b_val) {
        ++p;
      } else {
        ++q;
      }
      if (p >= fixedSize || q >= fixedSize) {
        break;
      }
    }
  }

  atomicAdd_sll(d_delta, IsAddition ? acc : -acc);
}

template <bool IsAddition>
__global__ void countHybridAnchorsKernel(
    const CBSTNode *__restrict__ d_v2vNodes, const int *__restrict__ d_v2vFlat,
    const CBSTNode *__restrict__ d_v2hNodes, const int *__restrict__ d_v2hFlat,
    const int *__restrict__ d_anchorIds, int anchorSize, int fixedSize,
    long long *__restrict__ d_delta) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= anchorSize) {
    return;
  }
  int u = d_anchorIds[tid];
  const CBSTNode *node_u = findNodeByKey(d_v2vNodes, u);
  if (node_u == nullptr) {
    return;
  }
  int loc_u = node_u->value;
  long long acc = 0;

  for (int off_v = loc_u; off_v < fixedSize; ++off_v) {
    int v = readFlat(d_v2vFlat, off_v);
    if (v == 0 || v == INT_MIN) {
      break;
    }
    if (v <= u) {
      continue;
    }

    const CBSTNode *node_v = findNodeByKey(d_v2vNodes, v);
    if (node_v == nullptr) {
      continue;
    }
    int loc_v = node_v->value;

    int p = loc_u;
    int q = loc_v;
    while (true) {
      int a_val = readFlat(d_v2vFlat, p);
      int b_val = readFlat(d_v2vFlat, q);
      if (a_val == 0 || a_val == INT_MIN || b_val == 0 || b_val == INT_MIN) {
        break;
      }
      if (a_val == b_val) {
        int w = a_val;
        if (w > v) {
          const CBSTNode *node_a = findNodeByKey(d_v2hNodes, u);
          const CBSTNode *node_b = findNodeByKey(d_v2hNodes, v);
          const CBSTNode *node_c = findNodeByKey(d_v2hNodes, w);
          if (node_a != nullptr && node_b != nullptr && node_c != nullptr) {
            int loc_a = node_a->value;
            int loc_b = node_b->value;
            int loc_c = node_c->value;
            if (group((int *)d_v2hFlat, loc_a, loc_b, loc_c) > 0) {
              ++acc;
            }
          }
        }
        ++p;
        ++q;
      } else if (a_val < b_val) {
        ++p;
      } else {
        ++q;
      }
      if (p >= fixedSize || q >= fixedSize) {
        break;
      }
    }
  }

  atomicAdd_sll(d_delta, IsAddition ? acc : -acc);
}

static long long copyAndReduceCounts(long long *d_counts, int count) {
  if (count <= 0) {
    return 0;
  }
  std::vector<long long> hostCounts(count, 0);
  checkCudaLocal(cudaMemcpy(hostCounts.data(), d_counts,
                            count * sizeof(long long), cudaMemcpyDeviceToHost));
  long long total = 0;
  for (long long value : hostCounts) {
    total += value;
  }
  return total;
}

static long long computeBaselineInner(const CBSTContext &h2vCtx,
                                      int numHyperedges) {
  if (numHyperedges <= 0) {
    return 0;
  }
  long long *d_counts = nullptr;
  checkCudaLocal(cudaMalloc(&d_counts, numHyperedges * sizeof(long long)));
  checkCudaLocal(cudaMemset(d_counts, 0, numHyperedges * sizeof(long long)));
  int block = 256;
  int grid = (numHyperedges + block - 1) / block;
  countInnerTrianglesKernel<<<grid, block>>>(
      h2vCtx.d_nodes, h2vCtx.d_flatPayload, numHyperedges, d_counts);
  checkCudaLocal(cudaDeviceSynchronize());
  long long total = copyAndReduceCounts(d_counts, numHyperedges);
  checkCudaLocal(cudaFree(d_counts));
  return total;
}

static long long computeBaselineHyperedge(const CBSTContext &h2hCtx,
                                          int numHyperedges) {
  if (numHyperedges <= 0) {
    return 0;
  }
  long long *d_counts = nullptr;
  checkCudaLocal(cudaMalloc(&d_counts, numHyperedges * sizeof(long long)));
  checkCudaLocal(cudaMemset(d_counts, 0, numHyperedges * sizeof(long long)));
  int block = 256;
  int grid = (numHyperedges + block - 1) / block;
  countHyperedgeTrianglesKernel<<<grid, block>>>(
      h2hCtx.d_nodes, h2hCtx.d_flatPayload, numHyperedges, h2hCtx.fixedSize,
      d_counts);
  checkCudaLocal(cudaDeviceSynchronize());
  long long total = copyAndReduceCounts(d_counts, numHyperedges);
  checkCudaLocal(cudaFree(d_counts));
  return total;
}

static long long computeBaselineOuter(const CBSTContext &v2hCtx,
                                      const CBSTContext &v2vCtx,
                                      int numVertices) {
  if (numVertices <= 0) {
    return 0;
  }
  long long *d_counts = nullptr;
  checkCudaLocal(cudaMalloc(&d_counts, numVertices * sizeof(long long)));
  checkCudaLocal(cudaMemset(d_counts, 0, numVertices * sizeof(long long)));
  int block = 256;
  int grid = (numVertices + block - 1) / block;
  countOuterTrianglesKernel<<<grid, block>>>(
      v2vCtx.d_nodes, v2vCtx.d_flatPayload, v2hCtx.d_nodes,
      v2hCtx.d_flatPayload, numVertices, v2vCtx.fixedSize, d_counts);
  checkCudaLocal(cudaDeviceSynchronize());
  long long total = copyAndReduceCounts(d_counts, numVertices);
  checkCudaLocal(cudaFree(d_counts));
  return total;
}

static long long computeBaselineHybrid(const CBSTContext &v2hCtx,
                                       const CBSTContext &v2vCtx,
                                       int numVertices) {
  if (numVertices <= 0) {
    return 0;
  }
  long long *d_counts = nullptr;
  checkCudaLocal(cudaMalloc(&d_counts, numVertices * sizeof(long long)));
  checkCudaLocal(cudaMemset(d_counts, 0, numVertices * sizeof(long long)));
  int block = 256;
  int grid = (numVertices + block - 1) / block;
  countHybridTrianglesKernel<<<grid, block>>>(
      v2vCtx.d_nodes, v2vCtx.d_flatPayload, v2hCtx.d_nodes,
      v2hCtx.d_flatPayload, numVertices, v2vCtx.fixedSize, d_counts);
  checkCudaLocal(cudaDeviceSynchronize());
  long long total = copyAndReduceCounts(d_counts, numVertices);
  checkCudaLocal(cudaFree(d_counts));
  return total;
}

static void innerSubtract(const CBSTContext &h2vCtx,
                          const std::vector<int> &deletedIds,
                          ScalarDeltaAccumulator &acc) {
  if (deletedIds.empty()) {
    return;
  }
  int *d_frontier = nullptr;
  checkCudaLocal(cudaMalloc(&d_frontier, deletedIds.size() * sizeof(int)));
  checkCudaLocal(cudaMemcpy(d_frontier, deletedIds.data(),
                            deletedIds.size() * sizeof(int),
                            cudaMemcpyHostToDevice));
  int block = 256;
  int grid = (static_cast<int>(deletedIds.size()) + block - 1) / block;
  innerFrontierKernel<false>
      <<<grid, block>>>(h2vCtx.d_nodes, h2vCtx.d_flatPayload, d_frontier,
                        static_cast<int>(deletedIds.size()), acc.d_value);
  checkCudaLocal(cudaDeviceSynchronize());
  checkCudaLocal(cudaFree(d_frontier));
}

static void innerAdd(const CBSTContext &h2vCtx,
                     const std::vector<int> &insertedIds,
                     ScalarDeltaAccumulator &acc) {
  if (insertedIds.empty()) {
    return;
  }
  int *d_frontier = nullptr;
  checkCudaLocal(cudaMalloc(&d_frontier, insertedIds.size() * sizeof(int)));
  checkCudaLocal(cudaMemcpy(d_frontier, insertedIds.data(),
                            insertedIds.size() * sizeof(int),
                            cudaMemcpyHostToDevice));
  int block = 256;
  int grid = (static_cast<int>(insertedIds.size()) + block - 1) / block;
  innerFrontierKernel<true>
      <<<grid, block>>>(h2vCtx.d_nodes, h2vCtx.d_flatPayload, d_frontier,
                        static_cast<int>(insertedIds.size()), acc.d_value);
  checkCudaLocal(cudaDeviceSynchronize());
  checkCudaLocal(cudaFree(d_frontier));
}

static void runHyperedgeAnchoredPhase(bool isAddition,
                                      const CBSTContext &h2hCtx,
                                      const std::vector<int> &frontier,
                                      long long *d_delta) {
  if (frontier.empty() || h2hCtx.numRecords <= 0) {
    return;
  }

  int *d_frontier = nullptr;
  checkCudaLocal(cudaMalloc(&d_frontier, frontier.size() * sizeof(int)));
  checkCudaLocal(cudaMemcpy(d_frontier, frontier.data(),
                            frontier.size() * sizeof(int),
                            cudaMemcpyHostToDevice));

  int flagsSize = h2hCtx.numRecords + 1;
  int *d_anchorFlags = nullptr;
  checkCudaLocal(cudaMalloc(&d_anchorFlags, flagsSize * sizeof(int)));
  checkCudaLocal(cudaMemset(d_anchorFlags, 0, flagsSize * sizeof(int)));

  int block = 256;
  int gridFrontier = (static_cast<int>(frontier.size()) + block - 1) / block;
  buildHyperedgeAnchorFlags<<<gridFrontier, block>>>(
      h2hCtx.d_nodes, h2hCtx.d_flatPayload, d_frontier,
      static_cast<int>(frontier.size()), h2hCtx.fixedSize, d_anchorFlags);
  checkCudaLocal(cudaDeviceSynchronize());

  std::vector<int> hostFlags(flagsSize, 0);
  checkCudaLocal(cudaMemcpy(hostFlags.data(), d_anchorFlags,
                            flagsSize * sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<int> anchors;
  anchors.reserve(frontier.size() * 4);
  for (int id = 1; id < flagsSize; ++id) {
    if (hostFlags[id] != 0) {
      anchors.push_back(id);
    }
  }

  if (!anchors.empty()) {
    int *d_anchors = nullptr;
    checkCudaLocal(cudaMalloc(&d_anchors, anchors.size() * sizeof(int)));
    checkCudaLocal(cudaMemcpy(d_anchors, anchors.data(),
                              anchors.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    int gridAnchors = (static_cast<int>(anchors.size()) + block - 1) / block;
    if (isAddition) {
      countHyperedgeAnchorsKernel<true><<<gridAnchors, block>>>(
          h2hCtx.d_nodes, h2hCtx.d_flatPayload, d_anchors,
          static_cast<int>(anchors.size()), h2hCtx.fixedSize, d_delta);
    } else {
      countHyperedgeAnchorsKernel<false><<<gridAnchors, block>>>(
          h2hCtx.d_nodes, h2hCtx.d_flatPayload, d_anchors,
          static_cast<int>(anchors.size()), h2hCtx.fixedSize, d_delta);
    }
    checkCudaLocal(cudaDeviceSynchronize());
    checkCudaLocal(cudaFree(d_anchors));
  }

  checkCudaLocal(cudaFree(d_anchorFlags));
  checkCudaLocal(cudaFree(d_frontier));
}

static void hyperedgeSubtract(const CBSTContext &h2hCtx,
                              const std::vector<int> &deletedIds,
                              ScalarDeltaAccumulator &acc) {
  runHyperedgeAnchoredPhase(false, h2hCtx, deletedIds, acc.d_value);
}

static void hyperedgeAdd(const CBSTContext &h2hCtx,
                         const std::vector<int> &insertedIds,
                         ScalarDeltaAccumulator &acc) {
  runHyperedgeAnchoredPhase(true, h2hCtx, insertedIds, acc.d_value);
}

static void runOuterAnchoredPhase(bool isAddition, const CBSTContext &v2hCtx,
                                  const CBSTContext &v2vCtx,
                                  const std::vector<int> &frontier,
                                  long long *d_delta) {
  if (frontier.empty() || v2vCtx.numRecords <= 0) {
    return;
  }

  int *d_frontier = nullptr;
  checkCudaLocal(cudaMalloc(&d_frontier, frontier.size() * sizeof(int)));
  checkCudaLocal(cudaMemcpy(d_frontier, frontier.data(),
                            frontier.size() * sizeof(int),
                            cudaMemcpyHostToDevice));

  int flagsSize = v2vCtx.numRecords + 1;
  int *d_anchorFlags = nullptr;
  checkCudaLocal(cudaMalloc(&d_anchorFlags, flagsSize * sizeof(int)));
  checkCudaLocal(cudaMemset(d_anchorFlags, 0, flagsSize * sizeof(int)));

  int block = 256;
  int gridFrontier = (static_cast<int>(frontier.size()) + block - 1) / block;
  buildOuterAnchorFlags<<<gridFrontier, block>>>(
      v2vCtx.d_nodes, v2vCtx.d_flatPayload, d_frontier,
      static_cast<int>(frontier.size()), v2vCtx.fixedSize, d_anchorFlags);
  checkCudaLocal(cudaDeviceSynchronize());

  std::vector<int> hostFlags(flagsSize, 0);
  checkCudaLocal(cudaMemcpy(hostFlags.data(), d_anchorFlags,
                            flagsSize * sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<int> anchors;
  anchors.reserve(frontier.size() * 4);
  for (int id = 1; id < flagsSize; ++id) {
    if (hostFlags[id] != 0) {
      anchors.push_back(id);
    }
  }

  if (!anchors.empty()) {
    int *d_anchors = nullptr;
    checkCudaLocal(cudaMalloc(&d_anchors, anchors.size() * sizeof(int)));
    checkCudaLocal(cudaMemcpy(d_anchors, anchors.data(),
                              anchors.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    int gridAnchors = (static_cast<int>(anchors.size()) + block - 1) / block;
    if (isAddition) {
      countOuterAnchorsKernel<true><<<gridAnchors, block>>>(
          v2vCtx.d_nodes, v2vCtx.d_flatPayload, v2hCtx.d_nodes,
          v2hCtx.d_flatPayload, d_anchors, static_cast<int>(anchors.size()),
          v2vCtx.fixedSize, d_delta);
    } else {
      countOuterAnchorsKernel<false><<<gridAnchors, block>>>(
          v2vCtx.d_nodes, v2vCtx.d_flatPayload, v2hCtx.d_nodes,
          v2hCtx.d_flatPayload, d_anchors, static_cast<int>(anchors.size()),
          v2vCtx.fixedSize, d_delta);
    }
    checkCudaLocal(cudaDeviceSynchronize());
    checkCudaLocal(cudaFree(d_anchors));
  }

  checkCudaLocal(cudaFree(d_anchorFlags));
  checkCudaLocal(cudaFree(d_frontier));
}

static void outerSubtract(const CBSTContext &v2hCtx, const CBSTContext &v2vCtx,
                          const std::vector<int> &deletedVertices,
                          ScalarDeltaAccumulator &acc) {
  runOuterAnchoredPhase(false, v2hCtx, v2vCtx, deletedVertices, acc.d_value);
}

static void outerAdd(const CBSTContext &v2hCtx, const CBSTContext &v2vCtx,
                     const std::vector<int> &insertedVertices,
                     ScalarDeltaAccumulator &acc) {
  runOuterAnchoredPhase(true, v2hCtx, v2vCtx, insertedVertices, acc.d_value);
}

static void runHybridAnchoredPhase(bool isAddition, const CBSTContext &v2hCtx,
                                   const CBSTContext &v2vCtx,
                                   const std::vector<int> &frontier,
                                   long long *d_delta) {
  if (frontier.empty() || v2vCtx.numRecords <= 0) {
    return;
  }

  int *d_frontier = nullptr;
  checkCudaLocal(cudaMalloc(&d_frontier, frontier.size() * sizeof(int)));
  checkCudaLocal(cudaMemcpy(d_frontier, frontier.data(),
                            frontier.size() * sizeof(int),
                            cudaMemcpyHostToDevice));

  int flagsSize = v2vCtx.numRecords + 1;
  int *d_anchorFlags = nullptr;
  checkCudaLocal(cudaMalloc(&d_anchorFlags, flagsSize * sizeof(int)));
  checkCudaLocal(cudaMemset(d_anchorFlags, 0, flagsSize * sizeof(int)));

  int block = 256;
  int gridFrontier = (static_cast<int>(frontier.size()) + block - 1) / block;
  buildOuterAnchorFlags<<<gridFrontier, block>>>(
      v2vCtx.d_nodes, v2vCtx.d_flatPayload, d_frontier,
      static_cast<int>(frontier.size()), v2vCtx.fixedSize, d_anchorFlags);
  checkCudaLocal(cudaDeviceSynchronize());

  std::vector<int> hostFlags(flagsSize, 0);
  checkCudaLocal(cudaMemcpy(hostFlags.data(), d_anchorFlags,
                            flagsSize * sizeof(int), cudaMemcpyDeviceToHost));
  std::vector<int> anchors;
  anchors.reserve(frontier.size() * 4);
  for (int id = 1; id < flagsSize; ++id) {
    if (hostFlags[id] != 0) {
      anchors.push_back(id);
    }
  }

  if (!anchors.empty()) {
    int *d_anchors = nullptr;
    checkCudaLocal(cudaMalloc(&d_anchors, anchors.size() * sizeof(int)));
    checkCudaLocal(cudaMemcpy(d_anchors, anchors.data(),
                              anchors.size() * sizeof(int),
                              cudaMemcpyHostToDevice));
    int gridAnchors = (static_cast<int>(anchors.size()) + block - 1) / block;
    if (isAddition) {
      countHybridAnchorsKernel<true><<<gridAnchors, block>>>(
          v2vCtx.d_nodes, v2vCtx.d_flatPayload, v2hCtx.d_nodes,
          v2hCtx.d_flatPayload, d_anchors, static_cast<int>(anchors.size()),
          v2vCtx.fixedSize, d_delta);
    } else {
      countHybridAnchorsKernel<false><<<gridAnchors, block>>>(
          v2vCtx.d_nodes, v2vCtx.d_flatPayload, v2hCtx.d_nodes,
          v2hCtx.d_flatPayload, d_anchors, static_cast<int>(anchors.size()),
          v2vCtx.fixedSize, d_delta);
    }
    checkCudaLocal(cudaDeviceSynchronize());
    checkCudaLocal(cudaFree(d_anchors));
  }

  checkCudaLocal(cudaFree(d_anchorFlags));
  checkCudaLocal(cudaFree(d_frontier));
}

static void hybridSubtract(const CBSTContext &v2hCtx, const CBSTContext &v2vCtx,
                           const std::vector<int> &deletedVertices,
                           ScalarDeltaAccumulator &acc) {
  runHybridAnchoredPhase(false, v2hCtx, v2vCtx, deletedVertices, acc.d_value);
}

static void hybridAdd(const CBSTContext &v2hCtx, const CBSTContext &v2vCtx,
                      const std::vector<int> &insertedVertices,
                      ScalarDeltaAccumulator &acc) {
  runHybridAnchoredPhase(true, v2hCtx, v2vCtx, insertedVertices, acc.d_value);
}

int main(int argc, char *argv[]) {
  HypergraphParams params;
  if (!parseCommandLineArgs(argc, argv, params)) {
    return 1;
  }
  printHypergraphParams(params);

  auto [hyperedgeToVertex, ignoredVertexToHyperedge] =
      generateHypergraph(params);
  (void)ignoredVertexToHyperedge;

  VertexSnapshots initialVertexSnapshots =
      buildVertexSnapshots(hyperedgeToVertex);
  std::vector<std::vector<int>> hyperedgeToHyperedge =
      hyperedgeAdjacency(initialVertexSnapshots.rawV2H, hyperedgeToVertex);

  auto [h2vFlatVertexIds, h2vStartOffsets] =
      flatten(hyperedgeToVertex, "Hyperedge to Vertex");
  auto [v2hTrimmedFlatIds, v2hTrimmedStartOffsets] =
      flatten(initialVertexSnapshots.trimmedV2H, "Trimmed Vertex to Hyperedge");
  auto [v2vTrimmedFlatIds, v2vTrimmedStartOffsets] =
      flatten(initialVertexSnapshots.trimmedV2V, "Trimmed Vertex to Vertex");
  auto [h2hFlatAdjacency, h2hStartOffsets] =
      flatten(hyperedgeToHyperedge, "Hyperedge to Hyperedge");

  auto [cbstH2VStartOffsets, cbstH2VKeys] = prepareCBSTData(h2vStartOffsets);
  auto [cbstV2HTrimmedStartOffsets, cbstV2HTrimmedKeys] =
      prepareCBSTData(v2hTrimmedStartOffsets);
  auto [cbstV2VTrimmedStartOffsets, cbstV2VTrimmedKeys] =
      prepareCBSTData(v2vTrimmedStartOffsets);
  auto [cbstH2HStartOffsets, cbstH2HKeys] = prepareCBSTData(h2hStartOffsets);

  int numVertices = static_cast<int>(initialVertexSnapshots.trimmedV2H.size());

  CBSTOperations h2vOps("H2V", params.payloadCapacity, params.alignment);
  h2vOps.construct(cbstH2VKeys, cbstH2VStartOffsets, params.numHyperedges,
                   h2vFlatVertexIds.data(),
                   static_cast<int>(h2vFlatVertexIds.size()));

  CBSTOperations v2hTrimmedOps("V2H-trimmed", params.payloadCapacity,
                               params.alignment);
  v2hTrimmedOps.construct(cbstV2HTrimmedKeys, cbstV2HTrimmedStartOffsets,
                          numVertices, v2hTrimmedFlatIds.data(),
                          static_cast<int>(v2hTrimmedFlatIds.size()));

  CBSTOperations v2vTrimmedOps("V2V-trimmed", params.payloadCapacity,
                               params.alignment);
  v2vTrimmedOps.construct(cbstV2VTrimmedKeys, cbstV2VTrimmedStartOffsets,
                          numVertices, v2vTrimmedFlatIds.data(),
                          static_cast<int>(v2vTrimmedFlatIds.size()));

  CBSTOperations h2hOps("H2H", params.payloadCapacity, params.alignment);
  h2hOps.construct(cbstH2HKeys, cbstH2HStartOffsets, params.numHyperedges,
                   h2hFlatAdjacency.data(),
                   static_cast<int>(h2hFlatAdjacency.size()));

  CoarseTriangleCounts baselineCounts;
  baselineCounts.inner =
      computeBaselineInner(h2vOps.context(), params.numHyperedges);
  baselineCounts.outer = computeBaselineOuter(
      v2hTrimmedOps.context(), v2vTrimmedOps.context(), numVertices);
  baselineCounts.hybrid = computeBaselineHybrid(
      v2hTrimmedOps.context(), v2vTrimmedOps.context(), numVertices);
  baselineCounts.hyperedge =
      computeBaselineHyperedge(h2hOps.context(), params.numHyperedges);

  std::cout << "Inner triangles: " << baselineCounts.inner << std::endl;
  std::cout << "Outer triangles: " << baselineCounts.outer << std::endl;
  std::cout << "Hybrid triangles: " << baselineCounts.hybrid << std::endl;
  std::cout << "Hyperedge triangles: " << baselineCounts.hyperedge << std::endl;

  int N = params.numHyperedges;
  int numDeletes = std::max(1, std::min(2, N));
  std::vector<int> deletedIds;
  for (int k = 0; k < numDeletes; ++k) {
    deletedIds.push_back(N - k);
  }
  std::sort(deletedIds.begin(), deletedIds.end());

  int numInserts = numDeletes + 1;
  auto generatedInserts =
      hyperedge2vertex(numInserts, params.maxVerticesPerHyperedge,
                       params.minVertexId, params.maxVertexId);
  int reuseK = std::min(numDeletes, numInserts);
  std::vector<int> insertAssignedIds(numInserts);
  for (int i = 0; i < reuseK; ++i) {
    insertAssignedIds[i] = deletedIds[i];
  }
  for (int i = reuseK; i < numInserts; ++i) {
    insertAssignedIds[i] = N + (i - reuseK) + 1;
  }

  std::vector<int> deletedVertices =
      collectVerticesFromHyperedges(hyperedgeToVertex, deletedIds);
  std::vector<int> insertedVertices = collectVerticesFromRows(generatedInserts);

  ScalarDeltaAccumulator innerAcc;
  ScalarDeltaAccumulator outerAcc;
  ScalarDeltaAccumulator hybridAcc;
  ScalarDeltaAccumulator hyperedgeAcc;
  innerAcc.init();
  outerAcc.init();
  hybridAcc.init();
  hyperedgeAcc.init();

  innerSubtract(h2vOps.context(), deletedIds, innerAcc);
  outerSubtract(v2hTrimmedOps.context(), v2vTrimmedOps.context(),
                deletedVertices, outerAcc);
  hybridSubtract(v2hTrimmedOps.context(), v2vTrimmedOps.context(),
                 deletedVertices, hybridAcc);
  hyperedgeSubtract(h2hOps.context(), deletedIds, hyperedgeAcc);

  h2vOps.erase(deletedIds);

  std::vector<int> h2vInsertKeys = insertAssignedIds;
  std::vector<int> h2vInsertPayload;
  std::vector<int> h2vInsertPrefix;
  for (const auto &hyperedge : generatedInserts) {
    for (int vertexId : hyperedge) {
      h2vInsertPayload.push_back(vertexId);
    }
    int prefix = (h2vInsertPrefix.empty() ? 0 : h2vInsertPrefix.back()) +
                 static_cast<int>(hyperedge.size());
    h2vInsertPrefix.push_back(prefix);
  }
  InsertMapping h2vMapping =
      h2vOps.insert(h2vInsertKeys, h2vInsertPayload, h2vInsertPrefix);

  int maxMappedKey = 0;
  for (int key : h2vMapping.itemToKey) {
    maxMappedKey = std::max(maxMappedKey, key);
  }
  int maxId = std::max(N, maxMappedKey);

  std::vector<std::vector<int>> updatedH2V = hyperedgeToVertex;
  if (static_cast<int>(updatedH2V.size()) < maxId) {
    updatedH2V.resize(maxId);
  }
  for (int deletedId : deletedIds) {
    if (deletedId >= 1 && deletedId <= static_cast<int>(updatedH2V.size())) {
      updatedH2V[deletedId - 1].clear();
    }
  }
  for (size_t i = 0; i < generatedInserts.size(); ++i) {
    int mappedKey = h2vMapping.itemToKey[i];
    if (mappedKey < 1) {
      continue;
    }
    if (mappedKey > static_cast<int>(updatedH2V.size())) {
      updatedH2V.resize(mappedKey);
    }
    updatedH2V[mappedKey - 1] = generatedInserts[i];
  }

  VertexSnapshots updatedVertexSnapshots = buildVertexSnapshots(updatedH2V);
  std::vector<std::vector<int>> updatedH2H =
      hyperedgeAdjacency(updatedVertexSnapshots.rawV2H, updatedH2V);

  auto [h2vFlatValsNew, h2vStartsNew] =
      flatten(updatedH2V, "Updated Hyperedge to Vertex");
  auto [v2hTrimmedFlatValsNew, v2hTrimmedStartsNew] = flatten(
      updatedVertexSnapshots.trimmedV2H, "Updated Trimmed Vertex to Hyperedge");
  auto [v2vTrimmedFlatValsNew, v2vTrimmedStartsNew] = flatten(
      updatedVertexSnapshots.trimmedV2V, "Updated Trimmed Vertex to Vertex");
  auto [h2hFlatValsNew, h2hStartsNew] =
      flatten(updatedH2H, "Updated Hyperedge to Hyperedge");

  auto [cbstH2VStartsNew, cbstH2VKeysNew] = prepareCBSTData(h2vStartsNew);
  auto [cbstV2HTrimmedStartsNew, cbstV2HTrimmedKeysNew] =
      prepareCBSTData(v2hTrimmedStartsNew);
  auto [cbstV2VTrimmedStartsNew, cbstV2VTrimmedKeysNew] =
      prepareCBSTData(v2vTrimmedStartsNew);
  auto [cbstH2HStartsNew, cbstH2HKeysNew] = prepareCBSTData(h2hStartsNew);

  int updatedNumVertices =
      static_cast<int>(updatedVertexSnapshots.trimmedV2H.size());

  CBSTOperations h2vOpsNew("H2V-new", params.payloadCapacity, params.alignment);
  h2vOpsNew.construct(cbstH2VKeysNew, cbstH2VStartsNew, maxId,
                      h2vFlatValsNew.data(),
                      static_cast<int>(h2vFlatValsNew.size()));

  CBSTOperations v2hTrimmedOpsNew("V2H-trimmed-new", params.payloadCapacity,
                                  params.alignment);
  v2hTrimmedOpsNew.construct(cbstV2HTrimmedKeysNew, cbstV2HTrimmedStartsNew,
                             updatedNumVertices, v2hTrimmedFlatValsNew.data(),
                             static_cast<int>(v2hTrimmedFlatValsNew.size()));

  CBSTOperations v2vTrimmedOpsNew("V2V-trimmed-new", params.payloadCapacity,
                                  params.alignment);
  v2vTrimmedOpsNew.construct(cbstV2VTrimmedKeysNew, cbstV2VTrimmedStartsNew,
                             updatedNumVertices, v2vTrimmedFlatValsNew.data(),
                             static_cast<int>(v2vTrimmedFlatValsNew.size()));

  CBSTOperations h2hOpsNew("H2H-new", params.payloadCapacity, params.alignment);
  h2hOpsNew.construct(cbstH2HKeysNew, cbstH2HStartsNew, maxId,
                      h2hFlatValsNew.data(),
                      static_cast<int>(h2hFlatValsNew.size()));

  std::vector<int> mappedInsertIds(h2vMapping.itemToKey.begin(),
                                   h2vMapping.itemToKey.end());
  innerAdd(h2vOpsNew.context(), mappedInsertIds, innerAcc);
  outerAdd(v2hTrimmedOpsNew.context(), v2vTrimmedOpsNew.context(),
           insertedVertices, outerAcc);
  hybridAdd(v2hTrimmedOpsNew.context(), v2vTrimmedOpsNew.context(),
            insertedVertices, hybridAcc);
  hyperedgeAdd(h2hOpsNew.context(), mappedInsertIds, hyperedgeAcc);

  std::cout << "Inner triangle delta: " << innerAcc.read() << std::endl;
  std::cout << "Outer triangle delta: " << outerAcc.read() << std::endl;
  std::cout << "Hybrid triangle delta: " << hybridAcc.read() << std::endl;
  std::cout << "Hyperedge triangle delta: " << hyperedgeAcc.read() << std::endl;

  delete[] cbstH2VKeys;
  delete[] cbstV2HTrimmedKeys;
  delete[] cbstV2VTrimmedKeys;
  delete[] cbstH2HKeys;
  delete[] cbstH2VKeysNew;
  delete[] cbstV2HTrimmedKeysNew;
  delete[] cbstV2VTrimmedKeysNew;
  delete[] cbstH2HKeysNew;

  return 0;
}
