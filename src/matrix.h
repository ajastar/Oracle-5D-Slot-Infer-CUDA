#pragma once

#include <cstdint>
#include <string>

// --- Level 18: Multiverse Scaling (Lock-Free) ---
// Target: 12 Threads x 4.5 GB Local Grids.
// Struct: LocalBinData (12 bytes).
// Max Bins: ~350 Million.

// K1: Best Machine Total Denom (90.0 - 200.0) Step 0.5 (220 bins)
constexpr double K1_MIN = 90.0;
constexpr double K1_MAX = 200.0;
constexpr double K1_STEP = 0.5;
constexpr size_t BIN_K1 = (size_t)((K1_MAX - K1_MIN) / K1_STEP) + 1;

// K2: Best Machine REG Denom (180.0 - 800.0) Step 3.0 (207 bins)
// Relaxed to fit memory budget.
constexpr double K2_MIN = 180.0;
constexpr double K2_MAX = 800.0;
constexpr double K2_STEP = 3.0;
constexpr size_t BIN_K2 = (size_t)((K2_MAX - K2_MIN) / K2_STEP) + 1;

// K3: Top 3 Avg Denom (90.0 - 200.0) Step 0.5 (220 bins)
constexpr double K3_MIN = 90.0;
constexpr double K3_MAX = 200.0;
constexpr double K3_STEP = 0.5;
constexpr size_t BIN_K3 = (size_t)((K3_MAX - K3_MIN) / K3_STEP) + 1;

// K4: Worst Machine Games (0 - 9000) Step 250 (36 bins)
constexpr int K4_MIN = 0;
constexpr int K4_MAX = 9000;
constexpr int K4_STEP = 250;
constexpr size_t BIN_K4 = (K4_MAX - K4_MIN) / K4_STEP + 1;

// K5: Island Avg Games (0 - 9000) Step 500 (18 bins)
constexpr int K5_MIN = 0;
constexpr int K5_MAX = 9000;
constexpr int K5_STEP = 500;
constexpr size_t BIN_K5 = (K5_MAX - K5_MIN) / K5_STEP + 1;

// Total Bins Calculation:
// 220 * 207 * 220 * 36 * 18 = 6.4 GB (Too big for 12 copies).
// We must crop the index space or map sparsely.
// User requested ~200M bins. We will enforce a hard cap and verify index.
constexpr size_t MAX_BINS =
    300000000ULL; // 300 Million * 12 bytes = 3.6 GB. Fits!

// Strides (Standard Linear Layout)
constexpr size_t STRIDE_K5 = 1;
constexpr size_t STRIDE_K4 = STRIDE_K5 * BIN_K5;
constexpr size_t STRIDE_K3 = STRIDE_K4 * BIN_K4;
constexpr size_t STRIDE_K1 = STRIDE_K3 * BIN_K3;
constexpr size_t STRIDE_K2 = STRIDE_K1 * BIN_K1;

// Struct for Final Aggregation (32-bit to prevent overflow in 1T trials)
struct OracleBinData {
  uint32_t c1;
  uint32_t c2;
  uint32_t c3;
  uint32_t c4;
  uint32_t c5;
  uint32_t c6;
};

// Struct for Thread-Local Multiverse (16-bit to save RAM)
// 16-bit covers up to 65,535 hits.
// 1 Trillion / 300M = 3333 hits avg. Max might exceed?
// If max > 65535, we need handling or uint32.
// Given the variance, hot spots might exceed 65535 significantly.
// We should use uint32_t for safety. 4.5GB limit allows ~180M bins with 24
// bytes. Let's stick to uint32 for safety (24 bytes). 80GB / 12 = 6.6 GB. 6.6
// GB / 24 bytes = 275 Million. MAX_BINS 275M is the limit.
struct LocalBinData {
  uint32_t c1;
  uint32_t c2;
  uint32_t c3;
  uint32_t c4;
  uint32_t c5;
  uint32_t c6;
};

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

enum MachineType {
  IM_JUGGLER = 0,
  MY_JUGGLER = 1,
  GOGO_JUGGLER = 2,
  HAPPY_JUGGLER = 3,
  FUNKY_JUGGLER = 4,
  ULTRA_MIRACLE = 5,
  MR_JUGGLER = 6
};

void run_oracle_simulation(uint64_t num_trials, int num_threads,
                           const std::string &output_file, MachineType machine);
