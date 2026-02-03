#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "cuda_matrix.h"
#include <cuda_runtime.h>

// Forward Declaration of Kernel Wrappers
template <MachineType M>
void launch_simulation(OracleBinData *d_grids, Xoroshiro128PlusState *d_states,
                       float *d_z_table, uint64_t trials_per_thread, int blocks,
                       int threads, int num_grids);

// Merge Kernel (Defined in cuda_kernels.cu)
__global__ void merge_grids_kernel(OracleBinData *dest, OracleBinData *src,
                                   size_t n);

// Kernel for RNG Init
__global__ void init_rng_kernel(Xoroshiro128PlusState *states, uint64_t seed,
                                int N);

constexpr int Z_TABLE_SIZE = 1 << 22;

void checkCuda(cudaError_t result, const char *func) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Error in " << func << ": " << cudaGetErrorString(result)
              << std::endl;
    exit(1);
  }
}

// Helper: Format large numbers with commas
std::string format_commas(uint64_t n) {
  std::string s = std::to_string(n);
  int insertPosition = s.length() - 3;
  while (insertPosition > 0) {
    s.insert(insertPosition, ",");
    insertPosition -= 3;
  }
  return s;
}

// Helper: Format Time (HH:MM:SS)
std::string format_time(double seconds) {
  int s = (int)seconds;
  int h = s / 3600;
  int m = (s % 3600) / 60;
  int sec = s % 60;
  std::ostringstream oss;
  oss << std::setfill('0') << std::setw(2) << h << ":" << std::setw(2) << m
      << ":" << std::setw(2) << sec;
  return oss.str();
}

int main(int argc, char **argv) {
  if (argc < 4) {
    std::cout << "Usage: cuda_profiler.exe [trials] [machine_code(my/im/etc)] "
                 "[outfile] [mode(overwrite/resume)]"
              << std::endl;
    return 1;
  }

  uint64_t total_trials = std::stoull(argv[1]);
  std::string machine_str = argv[2];
  std::string output_file = argv[3];

  MachineType m_type = IM_JUGGLER;
  if (machine_str == "my")
    m_type = MY_JUGGLER;
  else if (machine_str == "gogo")
    m_type = GOGO_JUGGLER;
  else if (machine_str == "happy")
    m_type = HAPPY_JUGGLER;
  else if (machine_str == "funky")
    m_type = FUNKY_JUGGLER;
  else if (machine_str == "ultra")
    m_type = ULTRA_MIRACLE;
  else if (machine_str == "mr")
    m_type = MR_JUGGLER;

  std::cout << "=== The Oracle 5D (CUDA RTX Port) ===" << std::endl;
  std::cout << "Target Hardware: RTX 5060 Ti (16GB VRAM)" << std::endl;
  std::cout << "Trials: " << format_commas(total_trials) << std::endl;

  // --- 1. Z-Table Init ---
  std::vector<float> h_z_table(Z_TABLE_SIZE);
  std::mt19937 gen(1234);
  std::normal_distribution<float> d(0.0f, 1.0f);
  for (int i = 0; i < Z_TABLE_SIZE; ++i)
    h_z_table[i] = d(gen);

  float *d_z_table;
  checkCuda(cudaMalloc(&d_z_table, Z_TABLE_SIZE * sizeof(float)),
            "Malloc Z-Table");
  checkCuda(cudaMemcpy(d_z_table, h_z_table.data(),
                       Z_TABLE_SIZE * sizeof(float), cudaMemcpyHostToDevice),
            "Copy Z-Table");

  // --- 2. Grid Allocation (Level 20: Dual Grid Optimization) ---
  // Single Grid: 7.2 GB
  // Dual Grid: 14.4 GB (Target for 16GB VRAM)
  size_t single_grid_bytes = MAX_BINS * sizeof(OracleBinData);
  size_t dual_grid_bytes = single_grid_bytes * 2;
  OracleBinData *d_grids;
  int num_grids = 1;

  std::cout << "Attempting Dual Grid Allocation ("
            << (double)dual_grid_bytes / 1024 / 1024 / 1024 << " GB)..."
            << std::endl;
  cudaError_t alloc_res = cudaMalloc(&d_grids, dual_grid_bytes);
  if (alloc_res == cudaSuccess) {
    std::cout
        << ">> SUCCESS: Dual Grid Enabled. Atomic contention reduced by ~50%."
        << std::endl;
    checkCuda(cudaMemset(d_grids, 0, dual_grid_bytes), "Memset Dual Grid");
    num_grids = 2;
  } else {
    std::cout << ">> WARNING: Dual Grid Alloc Failed ("
              << cudaGetErrorString(alloc_res) << "). Fallback to Single Grid."
              << std::endl;
    checkCuda(cudaMalloc(&d_grids, single_grid_bytes), "Malloc Single Grid");
    checkCuda(cudaMemset(d_grids, 0, single_grid_bytes), "Memset Single Grid");
    num_grids = 1;
  }

  // --- 2.5 Resume Mode (Data Loading) ---
  bool resume_mode = false;
  if (argc >= 5 && std::string(argv[4]) == "resume") {
    resume_mode = true;
  }

  uint64_t previous_trials = 0;
  if (resume_mode) {
    std::ifstream ifs(output_file, std::ios::binary);
    if (ifs) {
      OracleMeta meta;
      ifs.read(reinterpret_cast<char *>(&meta), sizeof(meta));

      if (meta.total_bins == MAX_BINS && meta.machine_type == (int)m_type) {
        previous_trials = meta.trials;
        std::cout << ">> RESUME MODE: Loading existing data ("
                  << format_commas(previous_trials) << " trials)..."
                  << std::endl;

        std::vector<OracleBinData> h_existing(MAX_BINS);
        ifs.read(reinterpret_cast<char *>(h_existing.data()),
                 single_grid_bytes);

        // Copy to Grid 0 (Dual Grid logic will merge others into this, so this
        // is safe base)
        checkCuda(cudaMemcpy(d_grids, h_existing.data(), single_grid_bytes,
                             cudaMemcpyHostToDevice),
                  "Upload Resume Data");
      } else {
        std::cout << ">> WARNING: Resume file mismatch (Bin/Machine type). "
                     "Starting fresh."
                  << std::endl;
      }
    } else {
      std::cout << ">> NOTE: No existing file to resume. Starting fresh."
                << std::endl;
    }
  }

  // --- 3. RNG States ---
  // Config: 512 Blocks * 512 Threads = 262,144 Threads.
  int blocks = 1024;
  int threads_per_block = 512;
  int total_threads = blocks * threads_per_block;

  Xoroshiro128PlusState *d_states;
  checkCuda(
      cudaMalloc(&d_states, total_threads * sizeof(Xoroshiro128PlusState)),
      "Malloc RNG");

  std::cout << "Initializing RNG States..." << std::flush;
  init_rng_kernel<<<blocks, threads_per_block>>>(d_states, 99999ULL,
                                                 total_threads);
  checkCuda(cudaDeviceSynchronize(), "Init RNG");
  std::cout << " Done." << std::endl;

  // --- 4. Simulation Loop ---
  auto start = std::chrono::high_resolution_clock::now();

  uint64_t trials_done = 0;
  // Batch Size
  uint64_t loops_per_batch =
      1; // TDR Mitigation: Reduce batch size to prevent timeouts
  uint64_t trials_per_batch = (uint64_t)total_threads * loops_per_batch;

  std::cout << std::endl; // Space before bar
  auto last_print = std::chrono::high_resolution_clock::now();

  while (trials_done < total_trials) {
    uint64_t remaining = total_trials - trials_done;
    uint64_t current_loops = loops_per_batch;
    if (remaining < trials_per_batch) {
      current_loops = remaining / total_threads;
      if (current_loops == 0)
        current_loops = 1;
    }

    switch (m_type) {
    case MY_JUGGLER:
      launch_simulation<MY_JUGGLER>(d_grids, d_states, d_z_table, current_loops,
                                    blocks, threads_per_block, num_grids);
      break;
    case GOGO_JUGGLER:
      launch_simulation<GOGO_JUGGLER>(d_grids, d_states, d_z_table,
                                      current_loops, blocks, threads_per_block,
                                      num_grids);
      break;
    case HAPPY_JUGGLER:
      launch_simulation<HAPPY_JUGGLER>(d_grids, d_states, d_z_table,
                                       current_loops, blocks, threads_per_block,
                                       num_grids);
      break;
    case FUNKY_JUGGLER:
      launch_simulation<FUNKY_JUGGLER>(d_grids, d_states, d_z_table,
                                       current_loops, blocks, threads_per_block,
                                       num_grids);
      break;
    case ULTRA_MIRACLE:
      launch_simulation<ULTRA_MIRACLE>(d_grids, d_states, d_z_table,
                                       current_loops, blocks, threads_per_block,
                                       num_grids);
      break;
    case MR_JUGGLER:
      launch_simulation<MR_JUGGLER>(d_grids, d_states, d_z_table, current_loops,
                                    blocks, threads_per_block, num_grids);
      break;
    default:
      launch_simulation<IM_JUGGLER>(d_grids, d_states, d_z_table, current_loops,
                                    blocks, threads_per_block, num_grids);
      break;
    }

    uint64_t batch_done = (uint64_t)total_threads * current_loops;
    trials_done += batch_done;

    // --- UX Polish (Level 21) ---
    auto now = std::chrono::high_resolution_clock::now();
    double time_since_print =
        std::chrono::duration<double>(now - last_print).count();

    // Update UI every 1.0 second or on completion
    if (time_since_print > 3.0 || trials_done >= total_trials) {
      last_print = now; // Reset timer

      double elapsed = std::chrono::duration<double>(now - start).count();
      double speed = 0.0;
      if (elapsed > 0)
        speed = (double)trials_done / elapsed;

      // Effective Games Speed: 1 Event ~ 140,000 Games
      double games_speed = speed * 140000.0;
      std::string g_speed_str;

      if (games_speed > 1e12) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << (games_speed / 1e12)
            << " T/s";
        g_speed_str = oss.str();
      } else {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << (games_speed / 1e9)
            << " G/s";
        g_speed_str = oss.str();
      }

      double progress = (double)trials_done / total_trials;
      double eta = 0.0;
      if (speed > 0)
        eta = (total_trials - trials_done) / speed;

      // Progress Bar
      int barWidth = 30;
      std::cout << "\r[";
      int pos = barWidth * progress;
      for (int i = 0; i < barWidth; ++i) {
        if (i < pos)
          std::cout << "=";
        else if (i == pos)
          std::cout << ">";
        else
          std::cout << " ";
      }
      std::cout << "] " << (int)(progress * 100.0) << "% | " << std::fixed
                << std::setprecision(2) << (speed / 1e6) << " M/s ("
                << g_speed_str << ") | "
                << "ETA: " << format_time(eta) << "  " << std::flush;
    }

    if (trials_done >= total_trials)
      break;
  }
  std::cout << std::endl << std::endl;

  // --- 5. Reduce Dual Grid (If Active) ---
  if (num_grids == 2) {
    std::cout << "Merging Dual Grids (14.4GB -> 7.2GB)..." << std::endl;
    OracleBinData *grid1 = d_grids;
    OracleBinData *grid2 = d_grids + MAX_BINS;

    // Launch Merge (Linear, massive parallelism)
    int merge_blocks = (MAX_BINS + 1023) / 1024;
    merge_grids_kernel<<<merge_blocks, 1024>>>(grid1, grid2, MAX_BINS);
    checkCuda(cudaDeviceSynchronize(), "Merge Grids");
  }

  // --- 6. Download & Save ---
  std::cout << "Downloading Result Grid..." << std::endl;
  std::vector<OracleBinData> h_grid(MAX_BINS);
  // We always download from d_grids[0]
  checkCuda(cudaMemcpy(h_grid.data(), d_grids, single_grid_bytes,
                       cudaMemcpyDeviceToHost),
            "Download Grid");

  std::cout << "Writing to " << output_file << "..." << std::endl;
  std::ofstream ofs(output_file, std::ios::binary);

  OracleMeta meta;
  meta.trials = previous_trials + trials_done;
  meta.total_bins = MAX_BINS;
  meta.bin_k1 = BIN_K1;
  meta.bin_k2 = BIN_K2;
  meta.bin_k3 = BIN_K3;
  meta.bin_k4 = BIN_K4;
  meta.bin_k5 = BIN_K5;
  meta.machine_type = (int)m_type;

  ofs.write(reinterpret_cast<const char *>(&meta), sizeof(meta));
  ofs.write(reinterpret_cast<const char *>(h_grid.data()), single_grid_bytes);

  // Cleanup
  cudaFree(d_grids);
  cudaFree(d_states);
  cudaFree(d_z_table);

  std::cout << "CUDA Simulation Complete." << std::endl;
  return 0;
}
