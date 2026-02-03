#include "island_profiler.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>


// Xoroshiro128+ implementation for speed
struct Xoroshiro128Plus {
  uint64_t s[2];

  Xoroshiro128Plus(uint64_t seed) {
    s[0] = seed;
    s[1] = seed + 0x9E3779B97F4A7C15; // Golden Ratio
    // Warm up
    next();
    next();
    next();
  }

  static inline uint64_t rotl(const uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
  }

  uint64_t next() {
    const uint64_t s0 = s[0];
    uint64_t s1 = s[1];
    const uint64_t result = s0 + s1;

    s1 ^= s0;
    s[0] = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
    s[1] = rotl(s1, 37);                   // c

    return result;
  }

  // Double in [0, 1)
  double next_double() { return (next() >> 11) * (1.0 / 9007199254740992.0); }

  // Int range [min, max]
  int next_int(int min, int max) { return min + (next() % (max - min + 1)); }
};

// --- Pattern Configuration ---
void setup_island(int pattern_id, int (&settings)[NUM_MACHINES],
                  int (&games)[NUM_MACHINES], Xoroshiro128Plus &rng) {
  // Defaults
  for (int i = 0; i < NUM_MACHINES; ++i) {
    settings[i] = 1;
    games[i] = 0;
  }

  switch (pattern_id) {
  case HELL_DEAD_SILENCE: // All 1, Low Traffic (500-1500)
    for (int i = 0; i < NUM_MACHINES; ++i)
      games[i] = rng.next_int(500, 1500);
    break;
  case HELL_SLAUGHTER: // All 1, High Traffic (7000-9000)
    for (int i = 0; i < NUM_MACHINES; ++i)
      games[i] = rng.next_int(7000, 9000);
    break;
  case HELL_ONE_TRAP: // 19x1 (High), 1x4 (High)
    for (int i = 0; i < NUM_MACHINES; ++i)
      games[i] = rng.next_int(6000, 9000);
    settings[rng.next() % NUM_MACHINES] = 4;
    break;
  case NORMAL_ACCIDENT: // All 1, Medium Traffic (3000-5000), looks lucky maybe?
    // Just simulate All 1 with medium traffic. Accidents happen stochastically.
    for (int i = 0; i < NUM_MACHINES; ++i)
      games[i] = rng.next_int(3000, 5000);
    break;
  case NORMAL_PLAYABLE: // All 2 (or mixed 2/3), Medium-High
    for (int i = 0; i < NUM_MACHINES; ++i) {
      settings[i] = 2; // Flat 2
      games[i] = rng.next_int(4000, 7000);
    }
    break;
  case NORMAL_SNIPER: // 1x6 (High), rest 1 (Low/Med)
    for (int i = 0; i < NUM_MACHINES; ++i) {
      settings[i] = 1;
      games[i] = rng.next_int(1000, 3000);
    }
    {
      int target = rng.next() % NUM_MACHINES;
      settings[target] = 6;
      games[target] = rng.next_int(8000, 9000); // Sniper hits it hard
    }
    break;
  case HEAVEN_HALF_HALF: // 10x1, 10x6. High traffic on 6s, mixed on 1s?
    // Assume recognized event -> High traffic overall
    for (int i = 0; i < NUM_MACHINES; ++i)
      games[i] = rng.next_int(5000, 9000);
    for (int i = 0; i < 10; ++i)
      settings[i] = 6; // First 10 are 6 (simplification, can shuffle)
    // Shuffle
    for (int i = NUM_MACHINES - 1; i > 0; i--) {
      int j = rng.next() % (i + 1);
      std::swap(settings[i], settings[j]);
    }
    break;
  case HEAVEN_ALL_HIGH: // 4/5/6 Mix. High Traffic.
    for (int i = 0; i < NUM_MACHINES; ++i) {
      int r = rng.next_int(0, 2);
      settings[i] = 4 + r; // 4, 5, 6
      games[i] = rng.next_int(6000, 9000);
    }
    break;
  case HEAVEN_CLUSTER: // 1/3 splits.
    for (int i = 0; i < NUM_MACHINES; ++i) {
      games[i] = rng.next_int(5000, 8000);
      if (i < 7)
        settings[i] = 4;
      else if (i < 14)
        settings[i] = 5;
      else
        settings[i] = 6;
    }
    break;
  }
}

// --- Batch Simulation ---
void run_simulation_batch(int num_sims, int pattern_id,
                          std::vector<BinData> &local_grid, uint64_t seed) {
  Xoroshiro128Plus rng(seed);

  // Pre-allocate arrays for machines
  int settings[NUM_MACHINES];
  int games[NUM_MACHINES];
  int bonus_counts[NUM_MACHINES]; // Total Bonuses

  for (int s = 0; s < num_sims; ++s) {
    setup_island(pattern_id, settings, games, rng);

    double best_denom = 9999.0;
    int worst_g = 99999;
    long long total_island_games = 0;

    // Simulate 20 machines
    for (int m = 0; m < NUM_MACHINES; ++m) {
      int g = games[m];
      total_island_games += g;
      if (g < worst_g)
        worst_g = g;

      int bb = 0;
      int rb = 0;
      double p_bb = SETTINGS[settings[m]].bb_prob;
      double p_rb = SETTINGS[settings[m]].rb_prob;
      double p_total = p_bb + p_rb;

      // Fast simulation: Iterate games? Too slow for 300M * 20 * 8000.
      // Use Inverse Transform Sampling or geometric distribution?
      // Geometric is faster for skipping games between bonuses.
      // Or just approximation if N is large?
      // For N=8000, iterating is roughly 8000 ops.
      // 300M * 20 * 5000 = 30,000 Billion ops. Too slow.
      // Optimization: Inverse geometric sampling.

      // However, we just need Total Bonuses.
      // Is it just Binomial(n, p)? Yes.
      // Can we sample Binomial efficiently?
      // For small p and large n, approx Normal or Poisson. But precise Binomial
      // sampling is better. Since we need "exact" feel, let's use
      // std::binomial_distribution equivalent or approximation. Std's binomial
      // is slow to construct. Normal Approximation: mean = n*p, stddev =
      // sqrt(n*p*(1-p)). Values are large enough (n ~ 8000, p ~ 1/130 -> count
      // ~ 60). Normal approx is excellent.

      // Let's use Normal Approx for speed.
      // Z = (counts - np) / sqrt(...)
      // counts = Z * sigma + mu

      // BB
      double mu_bb = g * p_bb;
      double sigma_bb = std::sqrt(g * p_bb * (1.0 - p_bb));
      // Box-Muller transform for simple Normal
      double r1 = rng.next_double();
      double r2 = rng.next_double();
      // protection
      if (r1 < 1e-9)
        r1 = 1e-9;
      double z1 =
          std::sqrt(-2.0 * std::log(r1)) * std::cos(2.0 * 3.1415926535 * r2);
      bb = std::round(mu_bb + z1 * sigma_bb);
      if (bb < 0)
        bb = 0;

      // RB
      double mu_rb = g * p_rb;
      double sigma_rb = std::sqrt(g * p_rb * (1.0 - p_rb));
      // Reuse r1/r2? Need new ones for independence.
      r1 = rng.next_double();
      if (r1 < 1e-9)
        r1 = 1e-9;
      r2 = rng.next_double();
      double z2 =
          std::sqrt(-2.0 * std::log(r1)) * std::cos(2.0 * 3.1415926535 * r2);
      rb = std::round(mu_rb + z2 * sigma_rb);
      if (rb < 0)
        rb = 0;

      int total_bonus = bb + rb;

      // Metric: Best Prob
      // Denom = Games / TotalBonus
      double denom = (total_bonus > 0) ? (double)g / total_bonus : 999.0;
      if (denom < best_denom)
        best_denom = denom;
    }

    double avg_games = (double)total_island_games / NUM_MACHINES;

    // Discretize
    // 1. Best Prob Denom
    // range MIN_DENOM(80) to MAX_DENOM(300)
    // val < 80 -> 0. val > 300 -> last.
    int idx_prob = 0;
    if (best_denom <= MIN_DENOM)
      idx_prob = 0;
    else if (best_denom >= MAX_DENOM)
      idx_prob = NUM_BINS_PROB - 1;
    else
      idx_prob = (int)((best_denom - MIN_DENOM) / DENOM_STEP);

    // 2. Worst Games
    int idx_worst = worst_g / GAME_STEP;
    if (idx_worst >= NUM_BINS_WORST)
      idx_worst = NUM_BINS_WORST - 1;

    // 3. Avg Games
    int idx_avg = (int)avg_games / GAME_STEP;
    if (idx_avg >= NUM_BINS_AVG)
      idx_avg = NUM_BINS_AVG - 1;

    // Flatten Index: [prob][worst][avg]
    // size = 50 * 20 * 20 = 20000
    int flat_idx = idx_prob * (NUM_BINS_WORST * NUM_BINS_AVG) +
                   idx_worst * NUM_BINS_AVG + idx_avg;

    local_grid[flat_idx].counts[pattern_id]++;
  }
}
