#!/bin/bash

echo "=== Dynamic Hypergraph Analysis Project Build Test ==="
echo

# Check if nvcc is available
if ! command -v nvcc &> /dev/null; then
    echo "❌ nvcc not found. Please ensure CUDA toolkit is installed."
    echo "   On cluster, run: module load cuda-toolkit/12.5"
    exit 1
fi

echo "✅ nvcc found: $(which nvcc)"

# Clean previous build
echo "🧹 Cleaning previous build..."
make clean

# Build the project
echo "🔨 Building project..."
if make; then
    echo "✅ Build successful!"
    echo
    echo "📁 Project structure:"
    echo "   ├── src/                   (Source code)"
    echo "   │   ├── main.cu           (Main CUDA implementation)"
    echo "   │   ├── graphGeneration.hpp (Header file)"
    echo "   │   └── graphGeneration.cpp (C++ implementation)"
    echo "   ├── build/                (Build artifacts)"
    echo "   │   └── main              (Executable)"
    echo "   ├── scripts/              (Build scripts)"
    echo "   ├── docs/                 (Documentation)"
    echo "   └── Makefile             (Build configuration)"
    echo
    echo "🚀 To run the program:"
    echo "   make run"
    echo "   or"
    echo "   ./build/main"
    echo
    echo "📋 Available make targets:"
    echo "   make all      - Build the project"
    echo "   make run      - Build and run"
    echo "   make clean    - Clean build artifacts"
    echo "   make rebuild  - Clean and rebuild"
    echo "   make help     - Show all targets"
else
    echo "❌ Build failed!"
    exit 1
fi
