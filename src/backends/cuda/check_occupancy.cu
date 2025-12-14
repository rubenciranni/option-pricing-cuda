#include <cuda.h>
#include <utility>

// Implementation of check_occupancy_function
// This is compiled separately to avoid multiple definition linker errors
std::pair<int, int> check_occupancy_function(CUfunction func) {
    int minGridSize = 30;
    int blockSize = 256; // Example block size, adjust as needed

    cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, reinterpret_cast<CUfunction>(func),
                                     nullptr, 0, 0);
    return {minGridSize, blockSize};
}
