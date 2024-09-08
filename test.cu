#include <vector>
#include <algorithm>
#include <iostream>

std::vector<std::vector<int>> alternate(const std::vector<std::vector<int>>& random2DVec) {
    // Step 1: Find the maximum value in random2DVec
    int maxValue = 0;
    for (const auto& row : random2DVec) {
        if (!row.empty()) {
            maxValue = std::max(maxValue, *std::max_element(row.begin(), row.end()));
        }
    }

    // Step 2: Initialize alter2DVec with size maxValue + 1 (to handle 0-indexing)
    std::vector<std::vector<int>> alter2DVec(maxValue + 1);

    // Step 3: Populate alter2DVec with indices from random2DVec
    for (int rowIndex = 0; rowIndex < random2DVec.size(); ++rowIndex) {
        for (int value : random2DVec[rowIndex]) {
            alter2DVec[value].push_back(rowIndex);  // Insert the row index at the position of the value
        }
    }

    return alter2DVec;
}

// Helper function to print 2D vector
void print2DVector(const std::vector<std::vector<int>>& vec2d) {
    for (int i = 0; i < vec2d.size(); ++i) {
        std::cout << "Index " << i << ": ";
        for (int val : vec2d[i]) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Example random2DVec
    std::vector<std::vector<int>> random2DVec = {
        {10, 20, 30},          // 0th row
        {10, 40, 50},              // 1st row
        {60, 70, 80, 90}       // 2nd row
    };

    // Call alternate function
    std::vector<std::vector<int>> alter2DVec = alternate(random2DVec);

    // Print the result
    print2DVector(alter2DVec);

    return 0;
}
