#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib> // For rand() and srand()
#include <ctime>   // For time()

// Predefined id_to_index array
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

// CUDA Kernel to compute motif index and count occurrences
__global__ void count_motif_kernel(int* deg_a, int* deg_b, int* deg_c, int* C_ab, int* C_bc, int* C_ca, int* g_abc, int* motif_counts, int n) {
    int tid = threadIdx.x;  // Each thread handles one motif_id (0 to 29)

    int count = 0;
    for (int i = 0; i < n; i++) {
        int a = deg_a[i] - (C_ab[i] + C_ca[i]) + g_abc[i];
        int b = deg_b[i] - (C_bc[i] + C_ab[i]) + g_abc[i];
        int c = deg_c[i] - (C_ca[i] + C_bc[i]) + g_abc[i];
        int d = C_ab[i] - g_abc[i];
        int e = C_bc[i] - g_abc[i];
        int f = C_ca[i] - g_abc[i];
        int g = g_abc[i];

        int motif_id = (a > 0) + ((b > 0) << 1) + ((c > 0) << 2) + ((d > 0) << 3) + ((e > 0) << 4) + ((f > 0) << 5) + ((g > 0) << 6);
        int index = id_to_index[motif_id] - 1;

        // Increment the count if the motif_id matches the thread's id
        if (index == tid) {
            count++;
        }
    }

    // Store the count in the result array at index tid
    motif_counts[tid] = count;
}

// Host function to call the CUDA kernel
void count_motif_parallel(int* deg_a, int* deg_b, int* deg_c, int* C_ab, int* C_bc, int* C_ca, int* g_abc, int* motif_counts, int n) {
    // Device pointers
    int *d_deg_a, *d_deg_b, *d_deg_c, *d_C_ab, *d_C_bc, *d_C_ca, *d_g_abc, *d_motif_counts;

    // Allocate device memory
    cudaMalloc((void**)&d_deg_a, n * sizeof(int));
    cudaMalloc((void**)&d_deg_b, n * sizeof(int));
    cudaMalloc((void**)&d_deg_c, n * sizeof(int));
    cudaMalloc((void**)&d_C_ab, n * sizeof(int));
    cudaMalloc((void**)&d_C_bc, n * sizeof(int));
    cudaMalloc((void**)&d_C_ca, n * sizeof(int));
    cudaMalloc((void**)&d_g_abc, n * sizeof(int));
    cudaMalloc((void**)&d_motif_counts, 30 * sizeof(int));  // 30 for values 0 to 29

    // Copy data from host to device
    cudaMemcpy(d_deg_a, deg_a, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_deg_b, deg_b, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_deg_c, deg_c, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_ab, C_ab, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_bc, C_bc, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_ca, C_ca, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g_abc, g_abc, n * sizeof(int), cudaMemcpyHostToDevice);

    // Initialize the motif_counts array on the device to 0
    cudaMemset(d_motif_counts, 0, 30 * sizeof(int));

    // Launch the CUDA kernel with exactly 30 threads (one thread for each motif id 0 to 29)
    count_motif_kernel<<<1, 30>>>(d_deg_a, d_deg_b, d_deg_c, d_C_ab, d_C_bc, d_C_ca, d_g_abc, d_motif_counts, n);

    // Copy the motif_counts back to the host
    cudaMemcpy(motif_counts, d_motif_counts, 30 * sizeof(int), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_deg_a);
    cudaFree(d_deg_b);
    cudaFree(d_deg_c);
    cudaFree(d_C_ab);
    cudaFree(d_C_bc);
    cudaFree(d_C_ca);
    cudaFree(d_g_abc);
    cudaFree(d_motif_counts);
}

// Function to randomly initialize data
void initialize_random_data(int* arr, int n, int min_val, int max_val) {
    for (int i = 0; i < n; i++) {
        arr[i] = min_val + rand() % (max_val - min_val + 1);
    }
}

int main() {
    int n = 1000; // Number of motifs to process
    srand(time(0)); // Seed for random number generation

    int *deg_a = new int[n];
    int *deg_b = new int[n];
    int *deg_c = new int[n];
    int *C_ab = new int[n];
    int *C_bc = new int[n];
    int *C_ca = new int[n];
    int *g_abc = new int[n];
    int *motif_counts = new int[30]; // To store counts for each motif type (0 to 29)

    // Initialize data randomly with values in a reasonable range
    initialize_random_data(deg_a, n, 1, 10); // degrees between 1 and 10
    initialize_random_data(deg_b, n, 1, 10);
    initialize_random_data(deg_c, n, 1, 10);
    initialize_random_data(C_ab, n, 0, 5);   // clustering coefficients between 0 and 5
    initialize_random_data(C_bc, n, 0, 5);
    initialize_random_data(C_ca, n, 0, 5);
    initialize_random_data(g_abc, n, 0, 2);  // g_abc values between 0 and 2

    // Call the parallel function to count motifs
    count_motif_parallel(deg_a, deg_b, deg_c, C_ab, C_bc, C_ca, g_abc, motif_counts, n);

    // Output or process the results (motif counts for values 0 to 29)
    for (int i = 0; i < 30; i++) {
        std::cout << "Motif ID " << i << ": " << motif_counts[i] << " occurrences" << std::endl;
    }

    // Clean up
    delete[] deg_a;
    delete[] deg_b;
    delete[] deg_c;
    delete[] C_ab;
    delete[] C_bc;
    delete[] C_ca;
    delete[] g_abc;
    delete[] motif_counts;

    return 0;
}
