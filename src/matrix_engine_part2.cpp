
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
