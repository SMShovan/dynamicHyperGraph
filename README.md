# Dynamic Hypergraph Analysis Project

A CUDA-based implementation for motif counting in dynamic hypergraphs using Complete Binary Search Trees and parallel processing.

## Quick Start

```bash
# Build the project
make

# Build all executables, including the paper-style coarse triangle counter
make all-types

# Run the program with default parameters
make run

# Run the paper-style coarse triangle counter
make run-coarse

# Run with custom parameters
./build/main <num_hyperedges> <max_vertices_per_hyperedge> <min_vertex_id> <max_vertex_id>

# Example: 10 hyperedges, up to 3 vertices each, vertex IDs 1-50
./build/main 10 3 1 50

# Clean build artifacts
make clean
```

## Project Structure

```
DynamicHypergraphMotif/
├── src/                    # Source code
│   ├── main.cu            # Main CUDA implementation
│   ├── graphGeneration.hpp # Graph generation header
│   └── graphGeneration.cpp # Graph generation implementation
├── build/                  # Build artifacts (auto-generated)
├── scripts/                # Build and utility scripts
├── docs/                   # Documentation
│   └── README.md          # Detailed documentation
├── Makefile               # Build configuration
└── .gitignore            # Git ignore file
```

## Documentation

For detailed documentation, see [docs/README.md](docs/README.md).

## Executables

- `build/main` - dynamic 30-bin hypergraph motif counting plus update delta flow
- `build/type1` - paper-style inner triangle count `sum_e C(|e|, 3)`
- `build/type2` - pairwise overlap motif count
- `build/type3` - three-hyperedge overlap motif count
- `build/coarseTriangle` - paper-style coarse counts for `inner`, `outer`, and `hyperedge` triangles, including exact deltas after the simulated update workflow

## Requirements

- CUDA Toolkit 12.5+
- NVIDIA GPU with CUDA support
- Thrust library (included with CUDA)

## License

This project is part of academic research in hypergraph analysis.
