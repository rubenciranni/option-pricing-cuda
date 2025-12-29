#include <cuda.h>

#include <utility>

std::pair<int, int> check_occupancy_function(CUfunction func) {
    int minGridSize = 30;
    int blockSize = 256;

    cuOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, reinterpret_cast<CUfunction>(func),
                                     nullptr, 0, 0);
    return {minGridSize, blockSize};
}
