#include <iostream>
#include <vector>
#include <algorithm>
#include <set>

// Function to find hyperedge adjacency
std::vector<std::vector<int>> hyperedgeAdjacency(
    const std::vector<std::vector<int>>& vertexToHyperedge, 
    const std::vector<std::vector<int>>& hyperedgeToVertex) {
    
    int nHyperedges = hyperedgeToVertex.size();
    
    // Resultant adjacency matrix for hyperedges
    std::vector<std::vector<int>> hyperedgeAdjacencyMatrix(nHyperedges);

    // Iterate through each hyperedge
    for (int hyperedge = 0; hyperedge < nHyperedges; ++hyperedge) {
        std::set<int> adjacentHyperedges;

        // Get the vertices connected by this hyperedge
        const std::vector<int>& vertices = hyperedgeToVertex[hyperedge];

        // For each vertex, find other hyperedges connected to it
        for (int vertex : vertices) {
            for (int otherHyperedge : vertexToHyperedge[vertex]) {
                if (otherHyperedge != hyperedge + 1) {  // Avoid self-loop
                    adjacentHyperedges.insert(otherHyperedge); // Ensure no duplicates
                }
            }
        }

        // Convert set to vector and store in adjacency matrix
        hyperedgeAdjacencyMatrix[hyperedge] = std::vector<int>(adjacentHyperedges.begin(), adjacentHyperedges.end());
    }

    return hyperedgeAdjacencyMatrix;
}

int main() {
    // Example from previous output

    std::vector<std::vector<int>> hyperedgeToVertex = {
        {3, 7, 2}, // Hyperedge 1
        {8, 9},    // Hyperedge 2
        {1, 5, 4}, // Hyperedge 3
        {2, 7, 9}, // Hyperedge 4
        {6, 3}     // Hyperedge 5
    };

    std::vector<std::vector<int>> vertexToHyperedge = {
        {},                       // Vertex 0 (not used)
        {3},                      // Vertex 1
        {1, 4},                   // Vertex 2
        {1, 5},                   // Vertex 3
        {3},                      // Vertex 4
        {3},                      // Vertex 5
        {5},                      // Vertex 6
        {1, 4},                   // Vertex 7
        {2},                      // Vertex 8
        {2, 4}                    // Vertex 9
    };

    // Call the function to find hyperedge adjacency
    std::vector<std::vector<int>> adjacencyMatrix = hyperedgeAdjacency(vertexToHyperedge, hyperedgeToVertex);

    // Print hyperedge adjacency matrix
    std::cout << "Hyperedge to Hyperedge Adjacency:" << std::endl;
    for (int i = 0; i < adjacencyMatrix.size(); ++i) {
        std::cout << "Hyperedge " << i + 1 << " is adjacent to: ";
        for (int adjHyperedge : adjacencyMatrix[i]) {
            std::cout << adjHyperedge << " ";
        }
        std::cout << std::endl;
    }

    return 0;
}
