#pragma once

#include <climits>

// Header-only device data and helpers to avoid cross-TU device linking
static __device__ int id_to_index[128] = {
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

// Follow back-pointer chains in a flat payload array.
// Negative values (other than INT_MIN) are chain pointers: jump to -val.
// On return, loc points to the resolved position.
static inline __device__ int readFlat(const int* flat, int& loc) {
    while (true) {
        int val = flat[loc];
        if (val < 0 && val != INT_MIN) {
            loc = -val;
            continue;
        }
        return val;
    }
}

static inline __device__ int deg(int* d_h2vFlatvalues, int loc) {
    int count = 0;
    while (true) {
        int val = readFlat(d_h2vFlatvalues, loc);
        if (val == 0 || val == INT_MIN) break;
        count++;
        loc++;
    }
    return count;
}

static inline __device__ int con(int* d_h2vFlatvalues, int loc_a, int loc_b) {
    int count = 0;
    int i = loc_a;
    int j = loc_b;
    while (true) {
        int va = readFlat(d_h2vFlatvalues, i);
        int vb = readFlat(d_h2vFlatvalues, j);
        if (va == INT_MIN || vb == INT_MIN || va == 0 || vb == 0) {
            break;
        }
        if (va == vb) {
            count++;
            i++;
            j++;
        } else if (va < vb) {
            i++;
        } else {
            j++;
        }
    }
    return count;
}

static inline __device__ int group(int* d_h2vFlatvalues, int loc_a, int loc_b, int loc_c) {
    int i = loc_a, j = loc_b, k = loc_c;
    int count = 0;
    while (true) {
        int va = readFlat(d_h2vFlatvalues, i);
        int vb = readFlat(d_h2vFlatvalues, j);
        int vc = readFlat(d_h2vFlatvalues, k);
        if (va == INT_MIN || va == 0 || vb == INT_MIN || vb == 0 || vc == INT_MIN || vc == 0) {
            break;
        }
        if (va == vb && vb == vc) {
            count++;
            i++;
            j++;
            k++;
        } else if (va < vb || va < vc) {
            i++;
        } else if (vb < va || vb < vc) {
            j++;
        } else {
            k++;
        }
    }
    return count;
}

// Inline device helper to compute motif index [0..29] from region counts
static inline __device__ int compute_motif_index(int deg_a, int deg_b, int deg_c,
                                                 int C_ab, int C_bc, int C_ca,
                                                 int g_abc) {
    int a = deg_a - (C_ab + C_ca) + g_abc;
    int b = deg_b - (C_bc + C_ab) + g_abc;
    int c = deg_c - (C_ca + C_bc) + g_abc;
    int d = C_ab - g_abc;
    int e = C_bc - g_abc;
    int f = C_ca - g_abc;
    int g = g_abc;

    int motif_id = (a > 0)
                 | ((b > 0) << 1)
                 | ((c > 0) << 2)
                 | ((d > 0) << 3)
                 | ((e > 0) << 4)
                 | ((f > 0) << 5)
                 | ((g > 0) << 6);
    int index = id_to_index[motif_id] - 1; // [0..29]
    return index;
}


