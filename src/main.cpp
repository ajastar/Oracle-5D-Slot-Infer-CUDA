#include "island_profiler.h"
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>


int main(int argc, char **argv) {
  int total_sims = 300000000; // 300M
  int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0)
    num_threads = 4;

  std::cout << "Starting Island Profiler..." << std::endl;
  std::cout << "Target Sims: " << total_sims << std::endl;
  std::cout << "Threads: " << num_threads << std::endl;

  // Total sims per pattern?
  // User wants "Simulate 300 million hall situations".
  // Does that mean 300M *TOTAL* across all patterns? Or 300M per pattern?
  // Assuming 300M Total for the "Cheat Sheet" to be comprehensive.
  // 300M / 12 patterns ~= 25M per pattern.
  // We have 9 patterns currently implemented.
  // Let's do 300M Total.

  int sims_per_pattern = total_sims / NUM_PATTERNS;

  // Grid Size
  // 50 x 20 x 20 = 20,000 bins.

  // We need to aggregate results from threads.
  // Each thread needs its own local grid to avoid locking.
  // 20,000 bins * 9 patterns * 4 bytes = 720KB per thread. perfectly fine.

  auto start_time = std::chrono::high_resolution_clock::now();

  std::vector<std::thread> threads;
  std::vector<std::vector<BinData>> thread_grids(num_threads);

  for (int t = 0; t < num_threads; ++t) {
    // Initialize grid
    thread_grids[t].resize(NUM_BINS_PROB * NUM_BINS_WORST * NUM_BINS_AVG);
    // Clear
    for (auto &b : thread_grids[t]) {
      for (int p = 0; p < NUM_PATTERNS; ++p)
        b.counts[p] = 0;
    }

    threads.emplace_back([t, num_threads, sims_per_pattern, &thread_grids]() {
      // Distribute patterns among threads?
      // Or have each thread run all patterns?
      // Better to split the WORK per pattern.

      for (int p = 0; p < NUM_PATTERNS; ++p) {
        // Each pattern needs 'sims_per_pattern' total.
        // Each thread does sims_per_pattern / num_threads
        int my_sims = sims_per_pattern / num_threads;
        run_simulation_batch(my_sims, p, thread_grids[t], 12345 + t + p * 999);
      }
    });
  }

  for (auto &th : threads) {
    th.join();
  }

  auto end_time = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> elapsed = end_time - start_time;
  std::cout << "Simulation finished in " << elapsed.count() << "s" << std::endl;

  // Merge grids
  std::cout << "Merging results..." << std::endl;
  std::vector<BinData> global_grid(NUM_BINS_PROB * NUM_BINS_WORST *
                                   NUM_BINS_AVG);
  // Init
  for (auto &b : global_grid)
    for (int p = 0; p < NUM_PATTERNS; ++p)
      b.counts[p] = 0;

  for (const auto &t_grid : thread_grids) {
    for (size_t i = 0; i < global_grid.size(); ++i) {
      for (int p = 0; p < NUM_PATTERNS; ++p) {
        global_grid[i].counts[p] += t_grid[i].counts[p];
      }
    }
  }

  save_to_json(global_grid, "hall_data.json");
  std::cout << "Done." << std::endl;

  return 0;
}
