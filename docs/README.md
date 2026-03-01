# Dynamic Hypergraph Analysis Project

A CUDA-based implementation for motif counting in dynamic hypergraphs using Complete Binary Search Trees and parallel processing.

## Project Overview

This project implements:
- **Hypergraph Data Structures**: Hyperedge-to-Vertex, Vertex-to-Hyperedge, and Hyperedge-to-Hyperedge mappings
- **Dynamic Data Management**: Complete Binary Search Trees on GPU for efficient insertion/deletion operations
- **Motif Counting**: Parallel counting of 30 different motif types in hypergraph triangles
- **Coarse Triangle Counting**: Paper-style `inner`, `outer`, and `hyperedge` triangle counts on GPU
- **GPU Acceleration**: CUDA kernels for high-performance computation

## Development Workflow

### Local Development + Cluster Execution

This project is designed for local development with cluster execution:

1. **Edit code locally** on your development machine
2. **Sync changes to cluster** using rsync
3. **Compile and run on cluster** with CUDA runtime

### Sync Commands

```bash
# Sync project to cluster
rsync -avz --progress /path/to/local/dynamicHyperGraph/ sskg8@mill.mst.edu:~/dynamicHyperGraph/

# Sync from cluster to local (if needed)
rsync -avz --progress sskg8@mill.mst.edu:~/dynamicHyperGraph/ /path/to/local/dynamicHyperGraph/
```

## Running on Cluster

### Load CUDA Module
```bash
module load cuda-toolkit/12.5
```

### Build and Run
```bash
nvcc main.cu -o main && ./main
```

## Project Structure

- `src/main.cu` - Main CUDA implementation with the 30-bin hypergraph motif pipeline
- `src/type1.cu` - Standalone inner-triangle executable
- `src/type2.cu` - Standalone pair-overlap executable
- `src/type3.cu` - Standalone three-hyperedge overlap executable
- `src/coarseTriangle.cu` - Standalone paper-style coarse triangle executable
- `README.md` - This file

## Counting Modes

- `src/HMotifCount.cu` and `src/main.cu` implement the repo's fine-grained 30-bin motif classification for hyperedge triangles.
- `src/coarseTriangle.cu` implements the paper's coarse taxonomy:
  - `inner`: triangles induced by one hyperedge
  - `outer`: vertex triangles in the pairwise graph that are not contained in any hyperedge
  - `hyperedge`: triangles in the hyperedge adjacency graph

## Requirements

- CUDA Toolkit 12.5+
- NVIDIA GPU with CUDA support
- Thrust library (included with CUDA)

## Features

- **Dynamic Updates**: Real-time insertion and deletion of hyperedges
- **Memory Efficient**: Flattened array representations with GPU-optimized padding
- **Parallel Processing**: CUDA kernels for motif counting and tree operations
- **Scalable**: Designed for large hypergraph datasets
