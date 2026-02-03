#include <iostream>
#include <vector>
#include <omp.h>
#include <algorithm>
#include <fstream>
#include <cmath>
#include <atomic>
#include <memory> 
#include <xmmintrin.h> 
#include <thread>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <random> 
#include <array>
#include <cstring>

#include "matrix.h"
#include "island_profiler.h"

// Force Architecture for Windows Headers
#ifdef _WIN32
#ifndef _AMD64_
#define _AMD64_
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <memoryapi.h>
// Fallback for older SDKs
#ifndef MEM_LARGE_PAGES
#define MEM_LARGE_PAGES 0x20000000
#endif
#else
#include <sys/mman.h>
#endif

// --- Optimization Level 19: Huge Page Allocator (TLB Optimization) ---
// reduces TLB misses for massive (>2GB) arrays.

template <typename T>
struct HugePageDeleter {
    void operator()(T* p) const {
        if (!p) return;
#ifdef _WIN32
        VirtualFree(p, 0, MEM_RELEASE);
#else
        free(p); 
#endif
    }
};

template <typename T>
std::unique_ptr<T[], HugePageDeleter<T>> allocate_huge(size_t count) {
    size_t bytes = count * sizeof(T);
    
    // Align to 2MB (Large Page Minimum)
    size_t large_page_min = 2 * 1024 * 1024;
    size_t aligned_bytes = (bytes + large_page_min - 1) & ~(large_page_min - 1);

    T* ptr = nullptr;
#ifdef _WIN32
    // Try Level 19: Huge Pages
    ptr = (T*)VirtualAlloc(NULL, aligned_bytes, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
    
    if (ptr) {
        // success
    } else {
        // Fallback to standard 4KB Pages
        ptr = (T*)VirtualAlloc(NULL, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    }
#else
    ptr = (T*)aligned_alloc(64, bytes);
#endif
    
    if (!ptr) throw std::bad_alloc();
    return std::unique_ptr<T[], HugePageDeleter<T>>(ptr);
}


// --- Level 18 Strategy ---

constexpr int Z_TABLE_SIZE = 1 << 20; 
constexpr int Z_MASK = Z_TABLE_SIZE - 1;
alignas(64) float g_z_table[Z_TABLE_SIZE];

void init_z_table() {
    std::cout << "Pre-calculating Z-Score Table (" << Z_TABLE_SIZE << " entries)..." << std::endl;
    std::mt19937 gen(5489u); 
    std::normal_distribution<float> d(0.0f, 1.0f);
    for(int i=0; i<Z_TABLE_SIZE; ++i) {
        g_z_table[i] = d(gen);
    }
}

// --- Compile-Time Constants ---
template<MachineType M> struct JugglerTraits;

template<> struct JugglerTraits<ULTRA_MIRACLE> {
    static constexpr double BB[7] = {0, 1.0/267.5, 1.0/261.1, 1.0/256.0, 1.0/242.7, 1.0/233.2, 1.0/216.3};
    static constexpr double RB[7] = {0, 1.0/425.6, 1.0/402.1, 1.0/350.5, 1.0/322.8, 1.0/297.9, 1.0/277.7};
};

template<> struct JugglerTraits<MR_JUGGLER> {
    static constexpr double BB[7] = {0, 1.0/268.6, 1.0/267.5, 1.0/260.1, 1.0/249.2, 1.0/240.9, 1.0/237.4};
    static constexpr double RB[7] = {0, 1.0/374.5, 1.0/354.2, 1.0/331.0, 1.0/291.3, 1.0/257.0, 1.0/237.4};
};

template<> struct JugglerTraits<MY_JUGGLER> {
    static constexpr double BB[7] = {0, 1.0/273.1, 1.0/270.8, 1.0/266.4, 1.0/254.0, 1.0/240.1, 1.0/229.1};
    static constexpr double RB[7] = {0, 1.0/409.6, 1.0/385.5, 1.0/336.1, 1.0/290.0, 1.0/268.6, 1.0/229.1};
};

template<> struct JugglerTraits<FUNKY_JUGGLER> {
    static constexpr double BB[7] = {0, 1.0/266.4, 1.0/259.0, 1.0/256.0, 1.0/249.2, 1.0/240.1, 1.0/219.9};
    static constexpr double RB[7] = {0, 1.0/439.8, 1.0/407.1, 1.0/366.1, 1.0/322.8, 1.0/299.3, 1.0/262.1};
};

template<> struct JugglerTraits<HAPPY_JUGGLER> {
    static constexpr double BB[7] = {0, 1.0/273.1, 1.0/270.8, 1.0/263.2, 1.0/254.0, 1.0/239.2, 1.0/226.0};
    static constexpr double RB[7] = {0, 1.0/397.2, 1.0/362.1, 1.0/332.7, 1.0/300.6, 1.0/273.1, 1.0/256.0};
};

template<> struct JugglerTraits<GOGO_JUGGLER> {
    static constexpr double BB[7] = {0, 1.0/259.0, 1.0/258.0, 1.0/257.0, 1.0/254.0, 1.0/247.3, 1.0/234.9};
    static constexpr double RB[7] = {0, 1.0/354.2, 1.0/332.7, 1.0/306.2, 1.0/268.6, 1.0/247.3, 1.0/234.9};
};

template<> struct JugglerTraits<IM_JUGGLER> {
    static constexpr double BB[7] = {0, 1.0/273.1, 1.0/269.7, 1.0/269.7, 1.0/259.0, 1.0/259.0, 1.0/255.0};
    static constexpr double RB[7] = {0, 1.0/439.8, 1.0/399.6, 1.0/331.0, 1.0/315.1, 1.0/255.0, 1.0/255.0};
};

struct Xoroshiro128Plus {
    uint64_t s[2];
    Xoroshiro128Plus(uint64_t seed) {
        s[0] = seed; s[1] = seed + 0x9E3779B97F4A7C15;
        next(); next(); next();
    }
    static inline uint64_t rotl(const uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }
    uint64_t next() {
        const uint64_t s0 = s[0]; uint64_t s1 = s[1];
        const uint64_t result = s0 + s1;
        s1 ^= s0; s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); s[1] = rotl(s1, 37);
        return result; 
    }
    double next_double() { return (next() >> 11) * (1.0 / 9007199254740992.0); }
    int next_int(int min, int max) { return min + (next() % (max - min + 1)); }
};

struct MachineResult {
    int total_games;
    int bb;
    int rb;
    double denom_total;
    double denom_rb;
};

template<double P_BB, double P_RB>
inline void sim_step_opt(int games_step, Xoroshiro128Plus& rng, int& bb_out, int& rb_out) {
    if (games_step <= 0) return;
    
    uint64_t r = rng.next();
    float z1 = g_z_table[r & Z_MASK];
    float z2 = g_z_table[(r >> 20) & Z_MASK];
    
    double mu_b = games_step * P_BB;
    const double sig_b = std::sqrt(mu_b * (1.0 - P_BB)); 
    
    int b = (int)(mu_b + z1 * sig_b + 0.5);
    if(b < 0) b = 0;
    
    double mu_r = games_step * P_RB;
    const double sig_r = std::sqrt(mu_r * (1.0 - P_RB));
    int r_cnt = (int)(mu_r + z2 * sig_r + 0.5);
    if(r_cnt < 0) r_cnt = 0;
    
    bb_out += b;
    rb_out += r_cnt;
}

// Player Logic Same Template
template<MachineType M, int S>
MachineResult simulate_player_behavior_tmpl(double traffic_level, Xoroshiro128Plus& rng) {
    constexpr double p_bb = JugglerTraits<M>::BB[S];
    constexpr double p_rb = JugglerTraits<M>::RB[S];
    
    double skill_factor = (traffic_level - 0.5) * 2.0; 
    if (skill_factor < 0) skill_factor = 0;
    
    double quit_denom_base = 300.0;
    double quit_denom_pro = 250.0;
    double active_quit_threshold = quit_denom_base - (quit_denom_base - quit_denom_pro) * skill_factor;

    double continue_denom_base = 250.0;
    double continue_denom_pro = 220.0;
    double active_continue_threshold = continue_denom_base - (continue_denom_base - continue_denom_pro) * skill_factor;

    int max_games = 7000 + (int)(rng.next_int(0, 2000) * traffic_level); 
    int current_g = 0;
    int total_bb = 0;
    int total_rb = 0;

    int step = 300;
    sim_step_opt<p_bb, p_rb>(step, rng, total_bb, total_rb);
    current_g += step;
    
    if ((total_bb + total_rb) == 0 && rng.next_double() < 0.30) goto END_PLAYER;

    step = 200;
    sim_step_opt<p_bb, p_rb>(step, rng, total_bb, total_rb);
    current_g += step;
    {
        int total_hits = total_bb + total_rb;
        bool bad = false;
        if (total_hits > 0) {
            bad = ((double)current_g > active_quit_threshold * (double)total_hits);
        } 
        if ((total_hits == 0 || bad) && rng.next_double() < 0.80) goto END_PLAYER;
    }

    while (current_g < max_games) {
        int total_hits = total_bb + total_rb;
        
        if (total_hits > 0) {
             if ((double)current_g < 140.0 * (double)total_hits) {
                 step = max_games - current_g;
                 sim_step_opt<p_bb, p_rb>(step, rng, total_bb, total_rb);
                 current_g += step;
                 break;
             }
        }

        step = 1000;
        if (current_g + step > max_games) step = max_games - current_g;
        sim_step_opt<p_bb, p_rb>(step, rng, total_bb, total_rb);
        current_g += step;
        
        total_hits = total_bb + total_rb;

        if (total_hits == 0) {
             if (rng.next_double() < 0.60) goto END_PLAYER;
        } else {
             if ((double)current_g > active_continue_threshold * (double)total_hits) {
                 if (rng.next_double() < 0.60) goto END_PLAYER;
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
        if (garbage && skill_factor > 0.8) resit_prob *= 0.1; 

        if (rng.next_double() < resit_prob) {
            int add = rng.next_int(1000, 2000);
            if (current_g + add > max_games) add = max_games - current_g;
            sim_step_opt<p_bb, p_rb>(add, rng, total_bb, total_rb);
            current_g += add;
        }
    }
    
    int total_hits = total_bb + total_rb;
    double d_total = (total_hits > 0) ? (double)current_g / total_hits : 9999.0;
    double d_rb = (total_rb > 0) ? (double)current_g / total_rb : 9999.0;
    return { current_g, total_bb, total_rb, d_total, d_rb };
}

inline size_t get_k4_index(int games) {
  size_t idx = games / K4_STEP;
  if (idx >= BIN_K4)
    idx = BIN_K4 - 1;
  return idx;
}

std::atomic<uint64_t> g_trials_done(0);
std::atomic<bool> g_running(true);
std::chrono::time_point<std::chrono::steady_clock> g_start_time;

void progress_monitor_thread(uint64_t total_trials) {
  while (g_running) {
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    if (!g_running)
      break;

    uint64_t current = g_trials_done.load(std::memory_order_relaxed);
    double pct = (double)current / total_trials * 100.0;

    auto now = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(now - g_start_time).count();
    double speed = (double)current / elapsed;

    uint64_t remaining =
        (total_trials > current) ? (total_trials - current) : 0;
    double eta_seconds = (speed > 0) ? (double)remaining / speed : 0.0;

    int eta_h = (int)(eta_seconds / 3600);
    int eta_m = (int)((eta_seconds - eta_h * 3600) / 60);
    int eta_s = (int)(eta_seconds) % 60;

    int width = 30;
    int pos = (int)(width * pct / 100.0);

    std::cout << "\r[";
    for (int i = 0; i < width; ++i) {
      if (i < pos)
        std::cout << "=";
      else if (i == pos)
        std::cout << ">";
      else
        std::cout << " ";
    }
    std::cout << "] " << std::fixed << std::setprecision(2) << pct << "%"
              << " | Speed: " << (speed / 1000000.0) << " M/sec"
              << " | ETA: " << std::setfill('0') << std::setw(2) << eta_h << ":"
              << std::setw(2) << eta_m << ":" << std::setw(2) << eta_s << "    "
              << std::flush;
  }
}

typedef MachineResult (*SimFunc)(double, Xoroshiro128Plus &);

template <MachineType M>
void run_oracle_impl(uint64_t num_trials, int num_threads,
                     const std::string &output_file) {
  SimFunc sim_table[7];
  sim_table[0] = simulate_player_behavior_tmpl<M, 1>;
  sim_table[1] = simulate_player_behavior_tmpl<M, 1>;
  sim_table[2] = simulate_player_behavior_tmpl<M, 2>;
  sim_table[3] = simulate_player_behavior_tmpl<M, 3>;
  sim_table[4] = simulate_player_behavior_tmpl<M, 4>;
  sim_table[5] = simulate_player_behavior_tmpl<M, 5>;
  sim_table[6] = simulate_player_behavior_tmpl<M, 6>;

  std::cout << "Grid Resolution: " << MAX_BINS
            << " bins (Mode: Level 19 - Huge Pages)." << std::endl;

  // --- Multiverse Allocation (Level 19: Huge Pages) ---
  std::vector<std::unique_ptr<LocalBinData[], HugePageDeleter<LocalBinData>>>
      universes(num_threads);
  try {
    for (int t = 0; t < num_threads; ++t) {
      volatile size_t vol_size = MAX_BINS; // Obfuscate
      universes[t] = allocate_huge<LocalBinData>(vol_size);
      std::memset(universes[t].get(), 0, vol_size * sizeof(LocalBinData));
    }
    std::cout << "Allocated " << num_threads
              << " universes with Huge/Large Pages if available." << std::endl;
    std::cout << "Total RAM reserved: "
              << (double)(num_threads * MAX_BINS * sizeof(LocalBinData)) /
                     1024 / 1024 / 1024
              << " GB" << std::endl;
  } catch (const std::bad_alloc &e) {
    std::cerr << "Memory Allocation Failed! Ask User to buy more RAM or check "
                 "Large Page Privilege."
              << std::endl;
    return;
  }

  g_start_time = std::chrono::steady_clock::now();
  g_trials_done.store(0);
  g_running.store(true);
  std::thread monitor(progress_monitor_thread, num_trials);

  omp_set_num_threads(num_threads);

  const int UPDATE_INTERVAL = 65536;

#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    uint64_t seed = 77777 + tid * 999999ULL;
    Xoroshiro128Plus rng(seed);
    LocalBinData *my_universe = universes[tid].get();

    int settings[20];
    MachineResult results[20];

    uint64_t local_progress = 0;

#pragma omp for schedule(dynamic, 65536)
    for (long long t = 0; t < num_trials; ++t) {

      local_progress++;
      if (local_progress >= UPDATE_INTERVAL) {
        g_trials_done.fetch_add(local_progress, std::memory_order_relaxed);
        local_progress = 0;
      }

      double scenario_roll = rng.next_double();
      double base_q, high_rate, traffic_T;

      if (scenario_roll < 0.70) {
        base_q = 1.0 + rng.next_double() * (2.2 - 1.0);
        high_rate = rng.next_double() * 0.15;
        traffic_T = 0.5 + rng.next_double() * 0.4;
      } else if (scenario_roll < 0.90) {
        base_q = 1.5 + rng.next_double() * (2.5 - 1.5);
        high_rate = 0.15 + rng.next_double() * 0.25;
        traffic_T = 0.7 + rng.next_double() * 0.3;
      } else {
        base_q = 3.0 + rng.next_double() * (3.0);
        high_rate = 0.50 + rng.next_double() * 0.50;
        traffic_T = 0.9 + rng.next_double() * 0.1;
      }

      int base_s_low = (int)base_q;
      double base_rem = base_q - base_s_low;
      int best_setting_truth = 0;

      for (int m = 0; m < 20; ++m) {
        int s;
        if (rng.next_double() < high_rate) {
          s = 4 + rng.next_int(0, 2);
        } else {
          s = base_s_low + (rng.next_double() < base_rem ? 1 : 0);
          if (s > 6)
            s = 6;
        }
        settings[m] = s;
        if (s > best_setting_truth)
          best_setting_truth = s;

        results[m] = sim_table[s](traffic_T, rng);
      }

      double best_total = 9999.0;
      double best_reg_of_best_total = 9999.0;
      int worst_g = 99999;
      long long total_island_g = 0;
      double min1 = 9999, min2 = 9999, min3 = 9999;

      for (int m = 0; m < 20; ++m) {
        total_island_g += results[m].total_games;
        if (results[m].total_games < worst_g)
          worst_g = results[m].total_games;
        double d = results[m].denom_total;
        if (d < best_total) {
          best_total = d;
          best_reg_of_best_total = results[m].denom_rb;
        }
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

      if (best_total < K1_MIN || best_total > K1_MAX)
        continue;
      if (best_reg_of_best_total < K2_MIN || best_reg_of_best_total > K2_MAX)
        continue;
      if (best_total >= best_reg_of_best_total)
        continue;

      size_t idx_k1 = (size_t)((best_total - K1_MIN) / K1_STEP);
      size_t idx_k2 = (size_t)((best_reg_of_best_total - K2_MIN) / K2_STEP);
      if (top3_avg < K3_MIN || top3_avg > K3_MAX)
        continue;
      size_t idx_k3 = (size_t)((top3_avg - K3_MIN) / K3_STEP);
      size_t idx_k4 = get_k4_index(worst_g);
      size_t idx_k5_raw = (size_t)(avg_g / K5_STEP);
      if (idx_k5_raw >= BIN_K5)
        idx_k5_raw = BIN_K5 - 1;
      size_t idx_k5 = idx_k5_raw;

      size_t flat_idx = idx_k2 * STRIDE_K2 + idx_k1 * STRIDE_K1 +
                        idx_k3 * STRIDE_K3 + idx_k4 * STRIDE_K4 +
                        idx_k5 * STRIDE_K5;

      if (flat_idx >= MAX_BINS)
        continue;

      switch (best_setting_truth) {
      case 1:
        my_universe[flat_idx].c1++;
        break;
      case 2:
        my_universe[flat_idx].c2++;
        break;
      case 3:
        my_universe[flat_idx].c3++;
        break;
      case 4:
        my_universe[flat_idx].c4++;
        break;
      case 5:
        my_universe[flat_idx].c5++;
        break;
      case 6:
        my_universe[flat_idx].c6++;
        break;
      }
    }

    if (local_progress > 0) {
      g_trials_done.fetch_add(local_progress, std::memory_order_relaxed);
    }
  }

  g_running.store(false);
  monitor.join();

  // --- Reduction Phase ---
  std::cout << "Collapsing Multiverse (Reducing to Single Grid)..."
            << std::endl;
  // Huge allocation for final grid as well
  volatile size_t globe_sz = MAX_BINS;
  auto final_grid = allocate_huge<OracleBinData>(globe_sz);
  std::memset(final_grid.get(), 0, globe_sz * sizeof(OracleBinData));

#pragma omp parallel for schedule(static)
  for (long long i = 0; i < (long long)MAX_BINS; ++i) {
    uint32_t s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
    for (int t = 0; t < num_threads; ++t) {
      s1 += universes[t][i].c1;
      s2 += universes[t][i].c2;
      s3 += universes[t][i].c3;
      s4 += universes[t][i].c4;
      s5 += universes[t][i].c5;
      s6 += universes[t][i].c6;
    }
    final_grid[i].c1 = s1;
    final_grid[i].c2 = s2;
    final_grid[i].c3 = s3;
    final_grid[i].c4 = s4;
    final_grid[i].c5 = s5;
    final_grid[i].c6 = s6;
  }

  std::cout << "Dumping Memory..." << std::endl;
  std::ofstream ofs(output_file, std::ios::binary);
  if (!ofs) {
    return;
  }

  OracleMeta meta;
  meta.trials = num_trials;
  meta.total_bins = MAX_BINS;
  meta.bin_k1 = BIN_K1;
  meta.bin_k2 = BIN_K2;
  meta.bin_k3 = BIN_K3;
  meta.bin_k4 = BIN_K4;
  meta.bin_k5 = BIN_K5;
  meta.machine_type = (int)M;
  ofs.write(reinterpret_cast<const char *>(&meta), sizeof(meta));
  ofs.write(reinterpret_cast<const char *>(final_grid.get()),
            MAX_BINS * sizeof(OracleBinData));

  std::cout << "Done." << std::endl;
}

void run_oracle_simulation(uint64_t num_trials, int num_threads,
                           const std::string &output_file,
                           MachineType machine_type) {
  std::cout << "Initializing The Oracle 5D (Last Level - Multiverse)..."
            << std::endl;
  init_z_table();

  switch (machine_type) {
  case ULTRA_MIRACLE:
    run_oracle_impl<ULTRA_MIRACLE>(num_trials, num_threads, output_file);
    break;
  case MR_JUGGLER:
    run_oracle_impl<MR_JUGGLER>(num_trials, num_threads, output_file);
    break;
  case MY_JUGGLER:
    run_oracle_impl<MY_JUGGLER>(num_trials, num_threads, output_file);
    break;
  case FUNKY_JUGGLER:
    run_oracle_impl<FUNKY_JUGGLER>(num_trials, num_threads, output_file);
    break;
  case HAPPY_JUGGLER:
    run_oracle_impl<HAPPY_JUGGLER>(num_trials, num_threads, output_file);
    break;
  case GOGO_JUGGLER:
    run_oracle_impl<GOGO_JUGGLER>(num_trials, num_threads, output_file);
    break;
  default:
    run_oracle_impl<IM_JUGGLER>(num_trials, num_threads, output_file);
    break;
  }
}
