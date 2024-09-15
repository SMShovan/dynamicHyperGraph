#include <iostream>
#include <vector>

int countCommonItems(const std::vector<int>& arr1, const std::vector<int>& arr2) {
    int i = 0, j = 0, count = 0;
    int len1 = arr1.size(), len2 = arr2.size();

    // Use a while loop to traverse both arrays
    while (i < len1 && j < len2) {
        if (arr1[i] == INT_MIN || arr2[j] == INT_MIN)
            break;
        if (arr1[i] == arr2[j]) {
            count++;  // Common item found
            i++;
            j++;
        } else if (arr1[i] < arr2[j]) {
            i++;  // Move pointer in arr1
        } else {
            j++;  // Move pointer in arr2
        }
    }

    return count;
}

int main() {
    std::vector<int> arr1 = {1, 2, 3, 4, 5};
    std::vector<int> arr2 = {3, 4, 5, 6, 7};

    std::cout << "Number of common items: " << countCommonItems(arr1, arr2) << std::endl;

    return 0;
}
