#pragma once

#include <cstdint>

// --- CUDA Shared Constants and Structures ---

// Use __host__ __device__ for shared structs
#ifdef __CUDACC__
#define HD __host__ __device__
#else
#define HD
#endif

// Grid Dimensions (Same as CPU Level 18)
constexpr double K1_MIN = 90.0;
constexpr double K1_MAX = 200.0;
constexpr double K1_STEP = 2.0; // Coarsened from 0.5 to fit memory
constexpr size_t BIN_K1 = (size_t)((K1_MAX - K1_MIN) / K1_STEP) + 1;

constexpr double K2_MIN = 180.0;
constexpr double K2_MAX = 800.0;
constexpr double K2_STEP = 10.0; // Coarsened from 3.0
constexpr size_t BIN_K2 = (size_t)((K2_MAX - K2_MIN) / K2_STEP) + 1;

constexpr double K3_MIN = 90.0;
constexpr double K3_MAX = 200.0;
constexpr double K3_STEP = 2.0; // Coarsened from 0.5
constexpr size_t BIN_K3 = (size_t)((K3_MAX - K3_MIN) / K3_STEP) + 1;

constexpr int K4_MIN = 0;
constexpr int K4_MAX = 9000;
constexpr int K4_STEP = 250; // Kept fine (User Request)
constexpr size_t BIN_K4 = (K4_MAX - K4_MIN) / K4_STEP + 1;

constexpr int K5_MIN = 0;
constexpr int K5_MAX = 9000;
constexpr int K5_STEP = 250; // Kept fine (User Request)
constexpr size_t BIN_K5 = (K5_MAX - K5_MIN) / K5_STEP + 1;

constexpr size_t MAX_BINS = 300000000ULL;

constexpr size_t STRIDE_K5 = 1;
constexpr size_t STRIDE_K4 = STRIDE_K5 * BIN_K5;
constexpr size_t STRIDE_K3 = STRIDE_K4 * BIN_K4;
constexpr size_t STRIDE_K1 = STRIDE_K3 * BIN_K3;
constexpr size_t STRIDE_K2 = STRIDE_K1 * BIN_K1;

// OracleBinData for CUDA (Must align to 32 bytes or handle coalesced access)
// We use simple struct. Atomicity is handled via atomicAdd on members or
// reinterpret_cast.
struct OracleBinData {
  uint32_t c1;
  uint32_t c2;
  uint32_t c3;
  uint32_t c4;
  uint32_t c5;
  uint32_t c6;
};

// 1KB is max random state size usually.
// Xoroshiro128+ is 16 bytes.
struct Xoroshiro128PlusState {
  uint64_t s0;
  uint64_t s1;
};

enum MachineType {
  IM_JUGGLER = 0,
  MY_JUGGLER = 1,
  GOGO_JUGGLER = 2,
  HAPPY_JUGGLER = 3,
  FUNKY_JUGGLER = 4,
  ULTRA_MIRACLE = 5,
  MR_JUGGLER = 6
};

// Output Metadata Structure
struct OracleMeta {
  uint64_t trials;
  size_t total_bins;
  size_t bin_k1;
  size_t bin_k2;
  size_t bin_k3;
  size_t bin_k4;
  size_t bin_k5;
  int machine_type;
};
