#include "kernels.cuh"
#include "device_utils.cuh"
#include <climits>

__global__ void computeNextMultipleOf4(int* partialSolution, int* tmp, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K)
    {
        int val = partialSolution[3*idx + 2];
        tmp[idx] = nextMultipleOf4(val);
    }
}

__global__ void updatePartialSolution(int* partialSolution, int* tmp, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < K)
    {
        partialSolution[3*idx + 2] = tmp[idx];
    }
}

__global__ void insertNode(CBSTNode* nodes, int* flatValues, int* insertIndices, int* insertValues, int* insertSizes, int insertSize, int* partialSolution) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < insertSize) {
        int insertIndex = insertIndices[tid];
        int* values;
        int numValues;
        if (tid == 0){
            values = insertValues;
            numValues = insertSizes[tid];
        }
        else{
            values = insertValues + insertSizes[tid - 1];
            numValues = insertSizes[tid] - insertSizes[tid - 1];
        }
        CBSTNode* current = nodes;
        while (current != nullptr && current->index != insertIndex) {
            if (current->index > insertIndex) {
                current = current->left;
            } else {
                current = current->right;
            }
        }
        if (current != nullptr) {
            int valueIndex = current->value;
            for (int i = 0; i < numValues; ++i) {
                bool isOverflow = false;
                while (true) {
                    int val = flatValues[valueIndex];
                    if (val == 0 || val == INT_MIN) break;
                    if (val < 0) { valueIndex = -val; continue; }
                    if (flatValues[valueIndex + 1] == INT_MIN) {
                        partialSolution[tid * 3] = valueIndex + 1;
                        partialSolution[tid * 3 + 1] = i;
                        partialSolution[tid * 3 + 2] = numValues - i;
                        isOverflow = true;
                        break;
                    }
                    valueIndex++;
                }
                if (isOverflow)
                    break;
                if (flatValues[valueIndex] != INT_MIN)
                    flatValues[valueIndex] = values[i];
            }
        }
    }
}

__global__ void allocateSpace(int* partialSolution, int* flatValues, int spaceAvailableFrom, int* insertIndices, int* insertValues, int* insertSizes, int insertSize){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < insertSize) {
        int* values;
        int numValues;
        if (tid == 0){
            values = insertValues;
            numValues = insertSizes[tid];
        }
        else{
            values = insertValues + insertSizes[tid - 1];
            numValues = insertSizes[tid] - insertSizes[tid - 1];
        }

        int idxPartialSolution = tid * 3;
        int startPartialSolution = idxPartialSolution + 1;
        int lenPartialSolution = idxPartialSolution + 2;

        if (tid == 0) {
            if (partialSolution[lenPartialSolution] == 0)
                return;
        } else {
            if (partialSolution[lenPartialSolution] == partialSolution[lenPartialSolution - 3])
                return;
        }

        int startIdx;
        int storeStartIdx;
        if (tid == 0)
        {
            startIdx = spaceAvailableFrom;

        }
        else
        {
            startIdx = spaceAvailableFrom + partialSolution[idxPartialSolution - 1];
        }

        storeStartIdx = startIdx;

        for (int i = partialSolution[startPartialSolution]; i < numValues; i++, startIdx++)
        {
            flatValues[startIdx] = values[i];
        }

        // Compute per-thread padded size (cumulative minus previous cumulative)
        int perThreadPadded = (tid == 0)
            ? partialSolution[lenPartialSolution]
            : partialSolution[lenPartialSolution] - partialSolution[lenPartialSolution - 3];

        // Zero gap between written data and sentinel
        int numWritten = numValues - partialSolution[startPartialSolution];
        for (int z = numWritten; z < perThreadPadded - 1; ++z)
            flatValues[storeStartIdx + z] = 0;

        // Sentinel at last position within padded space
        flatValues[storeStartIdx + perThreadPadded - 1] = INT_MIN;

        // Back-pointer from original chunk to this overflow location
        flatValues[partialSolution[idxPartialSolution]] = storeStartIdx * (-1);
    }
}


