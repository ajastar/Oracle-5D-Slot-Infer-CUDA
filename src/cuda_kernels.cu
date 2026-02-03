#include "cuda_matrix.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <stdio.h>

// --- Device Constants ---
// Z-Table in Global Memory (Read-Only Cache)
__device__ float *d_z_table_ptr;

template <MachineType M> struct JugglerTraitsGPU;

template <> struct JugglerTraitsGPU<ULTRA_MIRACLE> {
  static constexpr double BB[7] = {0,           1.0 / 267.5, 1.0 / 261.1,
                                   1.0 / 256.0, 1.0 / 242.7, 1.0 / 233.2,
                                   1.0 / 216.3};
  static constexpr double RB[7] = {0,           1.0 / 425.6, 1.0 / 402.1,
                                   1.0 / 350.5, 1.0 / 322.8, 1.0 / 297.9,
                                   1.0 / 277.7};
};
template <> struct JugglerTraitsGPU<MR_JUGGLER> {
  static constexpr double BB[7] = {0,           1.0 / 268.6, 1.0 / 267.5,
                                   1.0 / 260.1, 1.0 / 249.2, 1.0 / 240.9,
                                   1.0 / 237.4};
  static constexpr double RB[7] = {0,           1.0 / 374.5, 1.0 / 354.2,
                                   1.0 / 331.0, 1.0 / 291.3, 1.0 / 257.0,
                                   1.0 / 237.4};
};
template <> struct JugglerTraitsGPU<MY_JUGGLER> {
  static constexpr double BB[7] = {0,           1.0 / 273.1, 1.0 / 270.8,
                                   1.0 / 266.4, 1.0 / 254.0, 1.0 / 240.1,
                                   1.0 / 229.1};
  static constexpr double RB[7] = {0,           1.0 / 409.6, 1.0 / 385.5,
                                   1.0 / 336.1, 1.0 / 290.0, 1.0 / 268.6,
                                   1.0 / 229.1};
};
template <> struct JugglerTraitsGPU<FUNKY_JUGGLER> {
  static constexpr double BB[7] = {0,           1.0 / 266.4, 1.0 / 259.0,
                                   1.0 / 256.0, 1.0 / 249.2, 1.0 / 240.1,
                                   1.0 / 219.9};
  static constexpr double RB[7] = {0,           1.0 / 439.8, 1.0 / 407.1,
                                   1.0 / 366.1, 1.0 / 322.8, 1.0 / 299.3,
                                   1.0 / 262.1};
};
template <> struct JugglerTraitsGPU<HAPPY_JUGGLER> {
  static constexpr double BB[7] = {0,           1.0 / 273.1, 1.0 / 270.8,
                                   1.0 / 263.2, 1.0 / 254.0, 1.0 / 239.2,
                                   1.0 / 226.0};
  static constexpr double RB[7] = {0,           1.0 / 397.2, 1.0 / 362.1,
                                   1.0 / 332.7, 1.0 / 300.6, 1.0 / 273.1,
                                   1.0 / 256.0};
};
template <> struct JugglerTraitsGPU<GOGO_JUGGLER> {
  static constexpr double BB[7] = {0,           1.0 / 259.0, 1.0 / 258.0,
                                   1.0 / 257.0, 1.0 / 254.0, 1.0 / 247.3,
                                   1.0 / 234.9};
  static constexpr double RB[7] = {0,           1.0 / 354.2, 1.0 / 332.7,
                                   1.0 / 306.2, 1.0 / 268.6, 1.0 / 247.3,
                                   1.0 / 234.9};
};
template <> struct JugglerTraitsGPU<IM_JUGGLER> {
  static constexpr double BB[7] = {0,           1.0 / 273.1, 1.0 / 269.7,
                                   1.0 / 269.7, 1.0 / 259.0, 1.0 / 259.0,
                                   1.0 / 255.0};
  static constexpr double RB[7] = {0,           1.0 / 439.8, 1.0 / 399.6,
                                   1.0 / 331.0, 1.0 / 315.1, 1.0 / 255.0,
                                   1.0 / 255.0};
};

// --- RNG Device Function ---
__device__ inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}
__device__ inline uint64_t rng_next(uint64_t *s) {
  const uint64_t s0 = s[0];
  uint64_t s1 = s[1];
  const uint64_t result = s0 + s1;
  s1 ^= s0;
  s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16);
  s[1] = rotl(s1, 37);
  return result;
}
__device__ inline double rng_next_double(uint64_t *s) {
  return (rng_next(s) >> 11) * (1.0 / 9007199254740992.0);
}
__device__ inline int rng_next_int(uint64_t *s, int min, int max) {
  return min + (rng_next(s) % (max - min + 1));
}

// --- Init RNG Kernel ---
__global__ void init_rng_kernel(Xoroshiro128PlusState *states, uint64_t seed,
                                int N) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
    uint64_t s0 = seed + idx * 0x9E3779B97F4A7C15ULL;
    uint64_t s1 = s0 ^ 0xBF58476D1CE4E5B9ULL;
    // Warmup
    states[idx].s0 = s0;
    states[idx].s1 = s1;

    uint64_t temp_s[2] = {s0, s1};
    rng_next(temp_s);
    rng_next(temp_s);
    states[idx].s0 = temp_s[0];
    states[idx].s1 = temp_s[1];
  }
}

// --- Simulation Logic (Device) ---
constexpr int Z_MASK = (1 << 20) - 1;

// REFACTORED: No floating point templates
__device__ inline void sim_step_gpu(int games_step, uint64_t *rng_s,
                                    float *z_table, int &bb_out, int &rb_out,
                                    double P_BB, double P_RB) {
  if (games_step <= 0)
    return;

  uint64_t r = rng_next(rng_s);
  float z1 = z_table[r & Z_MASK];
  float z2 = z_table[(r >> 20) & Z_MASK];

  double mu_b = games_step * P_BB;
  const double sig_b = sqrt(mu_b * (1.0 - P_BB));

  int b = (int)(mu_b + z1 * sig_b + 0.5);
  if (b < 0)
    b = 0;

  double mu_r = games_step * P_RB;
  const double sig_r = sqrt(mu_r * (1.0 - P_RB));
  int r_cnt = (int)(mu_r + z2 * sig_r + 0.5);
  if (r_cnt < 0)
    r_cnt = 0;

  bb_out += b;
  rb_out += r_cnt;
}

struct MachineResultGPU {
  int total_games;
  int bb;
  int rb;
  double denom_total;
  double denom_rb;
};

template <MachineType M, int S>
__device__ MachineResultGPU simulate_player_behavior_gpu(double traffic_level,
                                                         uint64_t *rng_s,
                                                         float *z_table) {
  constexpr double p_bb = JugglerTraitsGPU<M>::BB[S];
  constexpr double p_rb = JugglerTraitsGPU<M>::RB[S];

  // Simplified logic for GPU (Branch divergence minimization is good but Level
  // 17 logic is robust)
  double skill_factor = (traffic_level - 0.5) * 2.0;
  if (skill_factor < 0)
    skill_factor = 0;

  double quit_denom_base = 300.0;
  double quit_denom_pro = 250.0;
  double active_quit_threshold =
      quit_denom_base - (quit_denom_base - quit_denom_pro) * skill_factor;

  double continue_denom_base = 250.0;
  double continue_denom_pro = 220.0;
  double active_continue_threshold =
      continue_denom_base -
      (continue_denom_base - continue_denom_pro) * skill_factor;

  int max_games = 7000 + (int)(rng_next_int(rng_s, 0, 2000) * traffic_level);
  int current_g = 0;
  int total_bb = 0;
  int total_rb = 0;

  int step = 300;
  // UPDATED CALL
  sim_step_gpu(step, rng_s, z_table, total_bb, total_rb, p_bb, p_rb);
  current_g += step;

  if ((total_bb + total_rb) == 0 && rng_next_double(rng_s) < 0.30)
    goto END_PLAYER;

  step = 200;
  sim_step_gpu(step, rng_s, z_table, total_bb, total_rb, p_bb, p_rb);
  current_g += step;
  {
    int total_hits = total_bb + total_rb;
    bool bad = false;
    if (total_hits > 0) {
      bad = ((double)current_g > active_quit_threshold * (double)total_hits);
    }
    if ((total_hits == 0 || bad) && rng_next_double(rng_s) < 0.80)
      goto END_PLAYER;
  }

  while (current_g < max_games) {
    int total_hits = total_bb + total_rb;
    if (total_hits > 0) {
      if ((double)current_g < 140.0 * (double)total_hits) {
        step = max_games - current_g;
        sim_step_gpu(step, rng_s, z_table, total_bb, total_rb, p_bb, p_rb);
        current_g += step;
        break;
      }
    }
    step = 1000;
    if (current_g + step > max_games)
      step = max_games - current_g;
    sim_step_gpu(step, rng_s, z_table, total_bb, total_rb, p_bb, p_rb);
    current_g += step;
    total_hits = total_bb + total_rb;
    if (total_hits == 0) {
      if (rng_next_double(rng_s) < 0.60)
        goto END_PLAYER;
    } else {
      if ((double)current_g > active_continue_threshold * (double)total_hits) {
        if (rng_next_double(rng_s) < 0.60)
          goto END_PLAYER;
      }
    }
  }
END_PLAYER:
  if (current_g < max_games) {
    int total_hits = total_bb + total_rb;
    bool garbage = true;
    if (total_hits > 0) {
      garbage = ((double)current_g > 300.0 * (double)total_hits);
    }
    double resit_prob = 0.10 * traffic_level;
    if (garbage && skill_factor > 0.8)
      resit_prob *= 0.1;
    if (rng_next_double(rng_s) < resit_prob) {
      int add = rng_next_int(rng_s, 1000, 2000);
      if (current_g + add > max_games)
        add = max_games - current_g;
      sim_step_gpu(add, rng_s, z_table, total_bb, total_rb, p_bb, p_rb);
      current_g += add;
    }
  }
  int total_hits = total_bb + total_rb;
  double d_total = (total_hits > 0) ? (double)current_g / total_hits : 9999.0;
  double d_rb = (total_rb > 0) ? (double)current_g / total_rb : 9999.0;
  MachineResultGPU r = {current_g, total_bb, total_rb, d_total, d_rb};
  return r;
}

// --- The Main Simulation Kernel ---
// Level 20 Update: Accepts num_grids for VRAM utilization
template <MachineType M>
__global__ void
simulation_kernel(OracleBinData *grids, Xoroshiro128PlusState *rng_states,
                  float *z_table, uint64_t trials_per_thread, int num_grids) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;

  // Select Sub-Grid based on Block ID
  // 5060 Ti can fit 2 grids.
  // We map blocks to grids via modulo or simply partition.
  // Modulo 2 is simplest: even blocks -> grid 0, odd blocks -> grid 1.
  // This halves collision probability.
  OracleBinData *my_grid = grids + (blockIdx.x % num_grids) * MAX_BINS;

  // Load RNG state to registers
  Xoroshiro128PlusState local_state = rng_states[tid];
  uint64_t rng_s[2] = {local_state.s0, local_state.s1};

  for (uint64_t t = 0; t < trials_per_thread; ++t) {

    // --- 1. Procedural Generation ---
    double scenario_roll = rng_next_double(rng_s);
    double base_q, high_rate, traffic_T;
    if (scenario_roll < 0.70) {
      base_q = 1.0 + rng_next_double(rng_s) * (2.2 - 1.0);
      high_rate = rng_next_double(rng_s) * 0.15;
      traffic_T = 0.5 + rng_next_double(rng_s) * 0.4;
    } else if (scenario_roll < 0.90) {
      base_q = 1.5 + rng_next_double(rng_s) * (2.5 - 1.5);
      high_rate = 0.15 + rng_next_double(rng_s) * 0.25;
      traffic_T = 0.7 + rng_next_double(rng_s) * 0.3;
    } else {
      base_q = 3.0 + rng_next_double(rng_s) * (3.0);
      high_rate = 0.50 + rng_next_double(rng_s) * 0.50;
      traffic_T = 0.9 + rng_next_double(rng_s) * 0.1;
    }

    int base_s_low = (int)base_q;
    double base_rem = base_q - base_s_low;

    // Track Settings for Candidate Verification
    int machine_settings[20];

    // Run 20 Machines
    MachineResultGPU results[20];

    for (int m = 0; m < 20; ++m) {
      int s;
      if (rng_next_double(rng_s) < high_rate) {
        s = 4 + rng_next_int(rng_s, 0, 2);
      } else {
        s = base_s_low + (rng_next_double(rng_s) < base_rem ? 1 : 0);
        if (s > 6)
          s = 6;
      }
      machine_settings[m] = s;

      // Dispatch Call
      switch (s) {
      case 1:
        results[m] =
            simulate_player_behavior_gpu<M, 1>(traffic_T, rng_s, z_table);
        break;
      case 2:
        results[m] =
            simulate_player_behavior_gpu<M, 2>(traffic_T, rng_s, z_table);
        break;
      case 3:
        results[m] =
            simulate_player_behavior_gpu<M, 3>(traffic_T, rng_s, z_table);
        break;
      case 4:
        results[m] =
            simulate_player_behavior_gpu<M, 4>(traffic_T, rng_s, z_table);
        break;
      case 5:
        results[m] =
            simulate_player_behavior_gpu<M, 5>(traffic_T, rng_s, z_table);
        break;
      case 6:
        results[m] =
            simulate_player_behavior_gpu<M, 6>(traffic_T, rng_s, z_table);
        break;
      }
    }

    // --- 2. Aggregation & Indexing ---
    // First, calculate Island Metrics (Top 3 Avg, Worst G, Avg G)
    double min1 = 9999.0, min2 = 9999.0, min3 = 9999.0;
    int worst_g = 99999;
    long long total_island_g = 0;

    for (int m = 0; m < 20; ++m) {
      double d = results[m].denom_total;
      total_island_g += results[m].total_games;
      if (results[m].total_games < worst_g)
        worst_g = results[m].total_games;

      if (d < min1) {
        min3 = min2;
        min2 = min1;
        min1 = d;
      } else if (d < min2) {
        min3 = min2;
        min2 = d;
      } else if (d < min3) {
        min3 = d;
      }
    }

    if (total_island_g == 0)
      continue;

    double avg_g = (double)total_island_g / 20.0;
    double top3_avg = (min1 + min2 + min3) / 3.0;

    // --- 3. Record ALL Machines (Level 23 Optimization) ---
    // Instead of filtering only the best, we loop again and record everyone.
    // This multiplies data collection speed by 20x and provides a perfect bell
    // curve.

    for (int m = 0; m < 20; ++m) {
      double d_total = results[m].denom_total;
      double d_reg_total =
          results[m].denom_rb; // denom of RB relative to total games? No,
                               // MachineResultGPU stores unique properties
      // Wait, machine result stores:
      // denom_total = total_games / total_bonus (1/X)
      // denom_rb = total_games / rb (1/Y)

      // Sanity check filters (Per Machine)
      if (d_total < K1_MIN || d_total > K1_MAX)
        continue;
      if (d_reg_total < K2_MIN || d_reg_total > K2_MAX)
        continue;
      if (d_total >= d_reg_total)
        continue; // Total Prob cannot be worse than RB Prob

      // Island Filters (Per Island - Global Context)
      if (top3_avg < K3_MIN || top3_avg > K3_MAX)
        continue;

      // Calculate Indices
      size_t idx_k1 = (size_t)((d_total - K1_MIN) / K1_STEP);
      size_t idx_k2 = (size_t)((d_reg_total - K2_MIN) / K2_STEP);
      size_t idx_k3 = (size_t)((top3_avg - K3_MIN) / K3_STEP);

      size_t idx_k4 = (size_t)(worst_g / K4_STEP);
      if (idx_k4 >= BIN_K4)
        idx_k4 = BIN_K4 - 1;

      size_t idx_k5 = (size_t)(avg_g / K5_STEP);
      if (idx_k5 >= BIN_K5)
        idx_k5 = BIN_K5 - 1;

      size_t flat_idx = idx_k2 * STRIDE_K2 + idx_k1 * STRIDE_K1 +
                        idx_k3 * STRIDE_K3 + idx_k4 * STRIDE_K4 +
                        idx_k5 * STRIDE_K5;

      // Atomic Update
      int target_setting = machine_settings[m];

      if (flat_idx < MAX_BINS) {
        switch (target_setting) {
        case 1:
          atomicAdd(&my_grid[flat_idx].c1, 1);
          break;
        case 2:
          atomicAdd(&my_grid[flat_idx].c2, 1);
          break;
        case 3:
          atomicAdd(&my_grid[flat_idx].c3, 1);
          break;
        case 4:
          atomicAdd(&my_grid[flat_idx].c4, 1);
          break;
        case 5:
          atomicAdd(&my_grid[flat_idx].c5, 1);
          break;
        case 6:
          atomicAdd(&my_grid[flat_idx].c6, 1);
          break;
        }
      }
    }
  }

  // Save RNG state
  rng_states[tid].s0 = rng_s[0];
  rng_states[tid].s1 = rng_s[1];
}

// --- Merge Kernel (Level 20) ---
__global__ void merge_grids_kernel(OracleBinData *dest, OracleBinData *src,
                                   size_t n) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    dest[idx].c1 += src[idx].c1;
    dest[idx].c2 += src[idx].c2;
    dest[idx].c3 += src[idx].c3;
    dest[idx].c4 += src[idx].c4;
    dest[idx].c5 += src[idx].c5;
    dest[idx].c6 += src[idx].c6;
  }
}

// Host Wrapper to Launch Kernel template
template <MachineType M>
void launch_simulation(OracleBinData *d_grids, Xoroshiro128PlusState *d_states,
                       float *d_z_table, uint64_t trials_per_thread, int blocks,
                       int threads, int num_grids) {
  simulation_kernel<M><<<blocks, threads>>>(d_grids, d_states, d_z_table,
                                            trials_per_thread, num_grids);
  cudaDeviceSynchronize();
}

// Explicit Instantiations
template void launch_simulation<IM_JUGGLER>(OracleBinData *,
                                            Xoroshiro128PlusState *, float *,
                                            uint64_t, int, int, int);
template void launch_simulation<MY_JUGGLER>(OracleBinData *,
                                            Xoroshiro128PlusState *, float *,
                                            uint64_t, int, int, int);
template void launch_simulation<GOGO_JUGGLER>(OracleBinData *,
                                              Xoroshiro128PlusState *, float *,
                                              uint64_t, int, int, int);
template void launch_simulation<HAPPY_JUGGLER>(OracleBinData *,
                                               Xoroshiro128PlusState *, float *,
                                               uint64_t, int, int, int);
template void launch_simulation<FUNKY_JUGGLER>(OracleBinData *,
                                               Xoroshiro128PlusState *, float *,
                                               uint64_t, int, int, int);
template void launch_simulation<ULTRA_MIRACLE>(OracleBinData *,
                                               Xoroshiro128PlusState *, float *,
                                               uint64_t, int, int, int);
template void launch_simulation<MR_JUGGLER>(OracleBinData *,
                                            Xoroshiro128PlusState *, float *,
                                            uint64_t, int, int, int);
