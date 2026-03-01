# Irregular-Graph Optimizations → DynamicHypergraph: Analysis and Mapping

This document summarizes the **Irregular-Graph-Optimizations-for-GPU** techniques, then maps them to the **dynamicHypergraph** project so we can design **thread-level**, **warp-level**, and **block-level** kernels for motif counting and CBST operations.

---

## Part 1: What the Irregular-Graph Project Teaches

### Workload (CSR neighbor gather)

- **Input**: CSR graph `rowPtr`, `colIdx`, vector `in[]`.
- **Output**: `out[v] = sum over (v→u) of in[u]` (SpMV-style gather).
- **Irregularity**: Row degrees vary wildly (power-law); a few hub rows dominate work.

### Technique taxonomy

| # | Name | Parallelism level | Idea |
|---|------|-------------------|------|
| **01** | Thread-per-row | **Thread** | One thread = one vertex/row. Simple; bad on skewed degrees (idle threads in warp). |
| **02** | Warp-per-row | **Warp** | One warp (32 threads) = one row. Lane-strided loop over edges; `warp_reduce_sum` for result. |
| **03** | Block-per-row | **Block** | One block = one *heavy* row. Block-strided loop; shared-memory reduction. |
| **04** | Degree binning | **Hybrid** | Split vertices into small / medium / large by degree; launch **thread** / **warp** / **block** kernels per bin. |
| **05** | Edge-parallel | **Edge** | One thread = one edge. Uniform work; **atomics** on `out[v]`. |
| **06** | Worklist queue | **Dynamic** | Global task list; threads `atomicAdd(globalHead,1)` to grab next vertex; process that row. |
| **07** | Scan compaction | **Active set** | Mark active vertices → exclusive scan → scatter compact list → process compact list (e.g. warp-per-row). |
| **08** | Persistent threads | **Dynamic** | Small grid; threads loop `atomicAdd(head,1)` until no tasks left (same pattern as 06, different grid size). |
| **09** | Two-phase heavy split | **Hybrid** | Light vertices → warp-per-row; heavy vertices → block-per-row (like 04 with 2 bins). |
| **10** | Hypergraph pin-parallel | **Pin/edge** | One thread = one pin (incidence entry); atomics per hyperedge (pattern for hypergraph). |
| **11** | Log binning | **Hybrid** | Bin by `floor(log2(degree))`; then thread / warp / block per bin (like 04 with log buckets). |

### Coding patterns (must-know)

1. **Thread-per-row**
   - `v = blockIdx.x * blockDim.x + threadIdx.x`
   - One thread, one row, sequential loop over row.

2. **Warp-per-row**
   - `warpId = tid >> 5`, `lane = threadIdx.x & 31`
   - One warp per row; loop `for (e = begin + lane; e < end; e += 32)`; then `warp_reduce_sum(sum)`; lane 0 writes.

3. **Block-per-row**
   - `idx = blockIdx.x` (one block = one row)
   - Loop `for (e = begin + threadIdx.x; e < end; e += blockDim.x)`; shared-memory reduction; thread 0 writes.

4. **Edge-parallel**
   - One thread per edge; need `rowId[e]` (row index for each edge); `atomicAdd(&out[v], in[u])`.

5. **Worklist / persistent**
   - `while (true) { i = atomicAdd(head, 1); if (i >= N) break; process(tasks[i]); }`

6. **Scan compaction**
   - Mark (e.g. `flags[v] = predicate(v)`), exclusive_scan(flags → positions), scatter active indices, then process the compact list (often warp-per-row).

7. **Warp reduction** (used in warp-per-row)
   - `warp_reduce_sum`: `for (offset = 16; offset > 0; offset >>= 1) x += __shfl_down_sync(0xffffffffu, x, offset);`

---

## Part 2: Current dynamicHypergraph Kernel Structure

### Data layout (reminder)

- **H2V, V2H, H2H**: Each is a CBST (nodes array + flat payload). Key = hyperedge or vertex ID; payload = variable-length list (with indirection in flat array).
- **Motif counting**: Triangles of *hyperedges* (i, j, k); work per hyperedge i = enumerate (i,j,k) and for each triangle compute degrees/intersections and bin into 30 motif types.

### Kernels and their current parallelism

| Component | Kernel(s) | Current pattern | Work unit |
|-----------|-----------|------------------|-----------|
| **Motif (full)** | `motifTriangleKernel` | **Thread-per-hyperedge** | One thread = one base hyperedge i; inner loops over j, then (j,k) intersection. |
| **Motif (delta)** | `motifFrontierKernel`, `motifAnchorKernel` | **Thread-per-frontier-item** | One thread = one frontier hyperedge; same triangle enumeration per thread. |
| **CBST build** | `buildEmptyBinaryTree`, `storeItemsIntoNodes` | **Thread-per-node** | One thread per tree node (regular). |
| **CBST find** | `findNode`, `findContents` | **Thread-per-query** | One thread per search key; BST walk per thread. |
| **CBST insert** | `insertNode`, `allocateSpace`, … | **Thread-per-insert-item** | One thread per new row; tree lookup + payload write. |
| **CBST delete** | `locateDeleteTargets`, `applyDeletes` | **Thread-per-delete-key** | One thread per key; BST walk then mark avail. |
| **CBST insert reuse** | `locateReusableSlots`, `extractSlotCapacities`, `lowerBoundKernel`, … | **Thread-per-item or per-slot** | Each kernel: one thread per logical item (K, D, M, etc.). |
| **CBST unfill** | `unfillKernel` | **Thread-per-key** | One thread per key; BST find + payload edits. |
| **CBST availability** | `reduceAvailLevel` | **Thread-per-node-in-level** | One thread per tree node in a level. |

### Where the imbalance is

- **Motif kernels**: Work per hyperedge i is proportional to (neighbors of i and their intersections). High-degree hyperedges (in H2H) do much more work → same **load imbalance** as in the CSR project (thread-per-row on power-law).
- **CBST lookups**: BST walk length varies by key (log n in expectation but variable). Slight imbalance; usually less dramatic than motif.

---

## Part 3: Applying Thread / Warp / Block Levels to dynamicHypergraph

### 3.1 Motif counting (main target)

**Current**: One thread per hyperedge (`i0 = blockIdx.x * blockDim.x + threadIdx.x`). Each thread iterates over H2H[i] neighbors j, then for each j intersects H2H[i] and H2H[j] to get k, then does H2V lookups and deg/con/group and atomics.

**Idea (mirror Irregular-Graph):**

- **Thread-level (baseline)**  
  Keep current: one thread = one hyperedge i. Easiest; good when degrees are small and similar.

- **Warp-level (warp-per-hyperedge)**  
  - One warp (32 threads) = one hyperedge i.
  - Warp cooperatively enumerates neighbors j (e.g. lane-strided over the H2H[i] list).
  - For each j, still need to intersect and count motif; options:
    - **Option A**: Still one “leader” (e.g. lane 0) does intersection and H2V work; other lanes help with neighbor enumeration only (reduce divergence in the outer loop).
    - **Option B**: Split triangles among lanes (e.g. each lane takes a subset of j’s), then warp-reduce partial motif counts (need 30-bin warp reduction or atomic per bin at the end).
  - Requires: mapping “global thread index” → (warpId → hyperedge i), and warp primitives for reduction (e.g. 30 ints: reduce in shared memory or use warp shuffle per bin).

- **Block-level (block-per-hyperedge)**  
  - One block = one *heavy* hyperedge i (e.g. high H2H degree).
  - Block-strided loop over j; for each j, one or more threads do intersection + motif computation; then block reduction into 30 bins (shared memory), final atomicAdd to global counts.
  - Use when a small fraction of hyperedges have very large neighbor sets.

- **Hybrid (degree binning / two-phase)**  
  - Classify hyperedges by H2H degree (or by “estimated work” if available):
    - **Light**: deg_H2H small → thread-per-hyperedge or warp-per-hyperedge.
    - **Heavy**: deg_H2H ≥ threshold → block-per-hyperedge.
  - Preprocessing: copy H2H degrees or row lengths to host (or compute on GPU), build lists of light/heavy hyperedges, then launch:
    - Kernel 1: light list → thread or warp per row.
    - Kernel 2: heavy list → block per row.
  - Same idea as technique_04 / technique_09 in Irregular-Graph.

- **Frontier (delta) motif**  
  - Already “active set” = deleted/inserted hyperedge IDs. Can keep thread-per-frontier-item; or use warp-per-frontier-item for frontier hyperedges with many triangles; or worklist/persistent if frontier is huge and work per item varies a lot.

### 3.2 CBST operations

- **Find / lookup**  
  - Currently thread-per-query. Optionally:
    - **Warp-per-query**: one warp per search key; lanes can help in tree walk only in limited ways (tree is pointer-chase). So thread-per-query is often fine; warp could help if you batch multiple keys per warp and coalesce memory.
  - **Block-per-query** is not natural for a single BST lookup (one traversal is one chain).

- **Insert (payload write)**  
  - One thread per insert item. Work = tree lookup + writing variable-length payload. If payloads are large, **warp-per-insert**: warp finds node, then lane-strided write of payload, then one lane does metadata. Similar to warp-per-row in CSR.

- **Delete**  
  - locateDeleteTargets: thread-per-key is fine (BST walk). applyDeletes: thread-per-key, very light. No strong need for warp/block unless we batch.

- **Insert reuse (best-fit, etc.)**  
  - Kernels are already “one thread per item” (slots, keys, etc.). Work is relatively uniform (binary search, scan). Degree-binning style doesn’t apply directly; persistent threads could help if we had a variable-sized task queue.

### 3.3 Summary table: where to use which level

| Area | Thread | Warp | Block | Hybrid / Other |
|------|--------|------|-------|-----------------|
| **Motif full count** | ✅ Current | ✅ Warp-per-hyperedge for medium degree | ✅ Block-per-hyperedge for hubs | Degree binning (light warp, heavy block) |
| **Motif delta** | ✅ Current | ✅ Optional warp-per-frontier-item | Optional for heavy frontier items | Worklist if frontier very large |
| **CBST find** | ✅ Keep | Optional coalesce/batch | — | — |
| **CBST insert payload** | ✅ Current | ✅ If payloads large | — | — |
| **CBST delete/reuse** | ✅ Keep | — | — | — |

---

## Part 4: Suggested implementation order for discussion

1. **Warp-per-hyperedge motif kernel (full count)**  
   - Redefine grid so that one warp = one hyperedge i.  
   - Lane-strided iteration over H2H[i] (neighbors j).  
   - Agree on how to assign intersection + deg/con/group work (one lane per j, or warp-reduce over j).  
   - Add warp (or shared-memory) reduction for 30 bins, then atomicAdd to global.

2. **Degree classification for hyperedges**  
   - Either from existing H2H flat payload (length per row) or from a pre-pass.  
   - Build `light_hyperedges[]` and `heavy_hyperedges[]`.  
   - Launch warp-per-row for light, block-per-row for heavy.

3. **Block-per-hyperedge for heavy rows**  
   - One block per heavy hyperedge; block-strided over j; shared memory for 30-bin partial counts; then atomicAdd to global.

4. **Delta motif**  
   - Keep current thread-per-frontier; optionally add warp-per-frontier-item for large frontiers.  
   - Scan compaction (technique 07) can compact “active” hyperedges if we ever have a sparse active set that isn’t already a small list.

5. **Pin-parallel / edge-parallel (technique 10 / 05)**  
   - For hypergraph, “pin-parallel” = one thread per (hyperedge, vertex) pair.  
   - In our project, that could apply to **building** or **traversing** H2V/V2H (e.g. parallel over pins for some aggregate). Motif counting itself is triangle-based, but auxiliary steps (e.g. computing degrees) could be pin-parallel with atomics.

---

## Part 5: References in code

- **Irregular-Graph**:  
  - Thread: `technique_01_thread_per_row.cu`  
  - Warp: `technique_02_warp_per_row.cu` (+ `warp_reduce_sum` in `cuda_common.cuh`)  
  - Block: `technique_03_block_per_row.cu`  
  - Binning / two-phase: `technique_04_degree_binning.cu`, `technique_09_two_phase_heavy_split.cu`  
  - Log binning: `technique_11_log_binning.cu`  
  - Edge / worklist / scan / persistent: techniques 05–08; hypergraph pattern: technique_10.

- **dynamicHypergraph**:  
  - Motif full: `src/HMotifCount.cu` (`motifTriangleKernel`)  
  - Motif delta: `src/HMotifCountUpdate.cu` (`motifFrontierKernel`, `motifAnchorKernel`)  
  - CBST: `kernel/build_tree.cu`, `kernel/find.cu`, `kernel/payload.cu`, `kernel/delete_avail.cu`, `kernel/insert_reuse.cu`, `structure/operations.cu`.

This document is the reference for designing and implementing **thread-level**, **warp-level**, and **block-level** kernels in the dynamicHypergraph project.
