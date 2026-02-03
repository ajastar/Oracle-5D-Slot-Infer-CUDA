#include "island_profiler.h"
#include <fstream>
#include <iostream>

void save_to_json(const std::vector<BinData> &grid,
                  const std::string &filename) {
  std::ofstream ofs(filename);
  ofs << "{\n";
  ofs << "  \"meta\": {\n";
  ofs << "    \"total_bins\": " << grid.size() << ",\n";
  ofs << "    \"patterns\": [\n";
  for (int i = 0; i < NUM_PATTERNS; ++i) {
    ofs << "      \"" << PATTERN_NAMES[i] << "\""
        << (i < NUM_PATTERNS - 1 ? "," : "") << "\n";
  }
  ofs << "    ]\n";
  ofs << "  },\n";
  ofs << "  \"bins\": [\n";

  bool first = true;
  int total_bins = NUM_BINS_PROB * NUM_BINS_WORST * NUM_BINS_AVG;

  for (int i = 0; i < total_bins; ++i) {
    // Optimization: Don't write empty bins
    bool empty = true;
    for (int p = 0; p < NUM_PATTERNS; ++p) {
      if (grid[i].counts[p] > 0) {
        empty = false;
        break;
      }
    }
    if (empty)
      continue;

    if (!first)
      ofs << ",\n";
    first = false;

    // Reconstruct indices
    int tmp = i;
    int idx_avg = tmp % NUM_BINS_AVG;
    tmp /= NUM_BINS_AVG;
    int idx_worst = tmp % NUM_BINS_WORST;
    tmp /= NUM_BINS_WORST;
    int idx_prob = tmp;

    // Reconstruct values (approx center of bin)
    double val_prob = MIN_DENOM + idx_prob * DENOM_STEP;
    int val_worst = idx_worst * GAME_STEP;
    int val_avg = idx_avg * GAME_STEP;

    ofs << "    {";
    ofs << "\"k_prob\": " << val_prob << ", ";
    ofs << "\"k_worst\": " << val_worst << ", ";
    ofs << "\"k_avg\": " << val_avg << ", ";
    ofs << "\"c\": [";
    for (int p = 0; p < NUM_PATTERNS; ++p) {
      ofs << grid[i].counts[p] << (p < NUM_PATTERNS - 1 ? "," : "");
    }
    ofs << "]}";
  }
  ofs << "\n  ]\n";
  ofs << "}\n";
  std::cout << "Data saved to " << filename << std::endl;
}
