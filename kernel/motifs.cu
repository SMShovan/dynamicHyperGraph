#include <climits>
#include <cstdio>

__device__ int id_to_index[128] = {
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0,
    21, 23, 22, 24, 23, 25, 24, 26,
    0, 0, 0, 0, 0, 0, 0, 0,
    21, 22, 23, 24, 23, 24, 25, 26,
    21, 23, 23, 25, 22, 24, 24, 26,
    27, 28, 28, 29, 28, 29, 29, 30,
    1, 2, 2, 3, 2, 3, 3, 4,
    5, 6, 6, 8, 7, 9, 9, 10,
    5, 7, 6, 9, 6, 9, 8, 10,
    11, 13, 12, 14, 13, 15, 14, 16,
    5, 6, 7, 9, 6, 8, 9, 10,
    11, 12, 13, 14, 13, 14, 15, 16,
    11, 13, 13, 15, 12, 14, 14, 16,
    17, 18, 18, 19, 18, 19, 19, 20
};

__device__ int deg(int* d_h2vFlatvalues, int loc) {
    int count = 0;
    while (d_h2vFlatvalues[loc] != 0 && d_h2vFlatvalues[loc] != INT_MIN )
    {
        count++;
        loc++;
    }
    return count;
}

__device__ int con(int* d_h2vFlatvalues, int loc_a, int loc_b) {
    int count = 0;
    int i = loc_a;
    int j = loc_b;
    while (true) {
        if (d_h2vFlatvalues[i] == INT_MIN || d_h2vFlatvalues[j] == INT_MIN || d_h2vFlatvalues[i] == 0 || d_h2vFlatvalues[j] == 0) {
            break;
        }
        if (d_h2vFlatvalues[i] == d_h2vFlatvalues[j]) {
            count++;
            i++;
            j++;
        } else if (d_h2vFlatvalues[i] < d_h2vFlatvalues[j]) {
            i++;
        } else {
            j++;
        }
    }
    return count;
}

__device__ int group(int* d_h2vFlatvalues, int loc_a, int loc_b, int loc_c) {
    int i = loc_a, j = loc_b, k = loc_c;
    int count = 0;
    while (true) {
        if (d_h2vFlatvalues[i] == INT_MIN || d_h2vFlatvalues[j] == INT_MIN || d_h2vFlatvalues[k] == INT_MIN ||
            d_h2vFlatvalues[i] == 0 || d_h2vFlatvalues[j] == 0 || d_h2vFlatvalues[k] == 0) {
            break;
        }
        if (d_h2vFlatvalues[i] == d_h2vFlatvalues[j] && d_h2vFlatvalues[j] == d_h2vFlatvalues[k]) {
            count++;
            i++;
            j++;
            k++;
        } else if (d_h2vFlatvalues[i] < d_h2vFlatvalues[j] || d_h2vFlatvalues[i] < d_h2vFlatvalues[k]) {
            i++;
        } else if (d_h2vFlatvalues[j] < d_h2vFlatvalues[i] || d_h2vFlatvalues[j] < d_h2vFlatvalues[k]) {
            j++;
        } else {
            k++;
        }
    }
    return count;
}


