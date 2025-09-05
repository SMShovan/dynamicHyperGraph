# Dynamic Hypergraph Analysis Project

A CUDA-based implementation for motif counting in dynamic hypergraphs using Complete Binary Search Trees and parallel processing.

## Quick Start

```bash
# Build the project
make

# Run the program with default parameters
make run

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

## Requirements

- CUDA Toolkit 12.5+
- NVIDIA GPU with CUDA support
- Thrust library (included with CUDA)

## License

This project is part of academic research in hypergraph analysis.