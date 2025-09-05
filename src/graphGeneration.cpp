#include "graphGeneration.hpp"
#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <algorithm>

std::vector<std::vector<int>> hyperedge2vertex(int n, int m, int r1, int r2) {
    std::vector<std::vector<int>> vec2d(n);
    std::srand(std::time(0)); // Seed for random number generation

    for (int i = 0; i < n; ++i) {
        int innerSize = rand() % m + 1; // Random inner size from 1 to m
        vec2d[i].resize(innerSize);
        for (int j = 0; j < innerSize; ++j) {
            vec2d[i][j] = rand() % (r2 - r1 + 1) + r1; // Random value in range [r1, r2]
        }
    }

    return vec2d;
}

std::vector<std::vector<int>> vertex2hyperedge(const std::vector<std::vector<int>>& hyperedgeToVertex) {
    // Step 1: Find the maximum value in hyperedgeToVertex
    int maxValue = 0;
    for (const auto& row : hyperedgeToVertex) {
        if (!row.empty()) {
            maxValue = std::max(maxValue, *std::max_element(row.begin(), row.end()));
        }
    }

    // Step 2: Initialize vertexToHyperedge with size maxValue + 1 (to handle 0-indexing)
    std::vector<std::vector<int>> vertexToHyperedge(maxValue + 1);

    // Step 3: Populate vertexToHyperedge with indices from hyperedgeToVertex
    for (int rowIndex = 0; rowIndex < hyperedgeToVertex.size(); ++rowIndex) {
        for (int value : hyperedgeToVertex[rowIndex]) {
            vertexToHyperedge[value].push_back(rowIndex + 1);  // Insert the row index at the position of the value
        }
    }

    return vertexToHyperedge;
}

void print2DVector(const std::vector<std::vector<int>>& vec2d) {
    std::cout << "2D Vector (Matrix Form):" << std::endl;
    for (const auto& row : vec2d) {
        for (int val : row) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}
