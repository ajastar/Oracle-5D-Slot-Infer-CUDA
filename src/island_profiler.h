#pragma once

#include <vector>
#include <string>
#include <cstdint>
#include <array>
#include <random>

// --- Constants ---
constexpr int NUM_MACHINES = 20;
constexpr int NUM_BINS_PROB = 50;
constexpr int NUM_BINS_WORST = 20;
constexpr int NUM_BINS_AVG = 20;
constexpr int NUM_PATTERNS = 9; // Currently 9 based on list, can expand to 12

// --- Juggler Settings (I'm Juggler EX) ---
struct SettingResult {
    double bb_prob;
    double rb_prob;
};

// 1-6
const SettingResult SETTINGS[] = {
    {0, 0}, // 0 index unused
    {1.0/287.4, 1.0/455.1}, // Setting 1
    {1.0/282.5, 1.0/409.6}, // Setting 2
    {1.0/282.5, 1.0/348.6}, // Setting 3
    {1.0/273.1, 1.0/321.3}, // Setting 4
    {1.0/255.0, 1.0/268.6}, // Setting 5
    {1.0/255.0, 1.0/255.0}  // Setting 6
};

// --- Patterns ---
enum PatternType {
    HELL_DEAD_SILENCE = 0,
    HELL_SLAUGHTER,
    HELL_ONE_TRAP,
    NORMAL_ACCIDENT,
    NORMAL_PLAYABLE,
    NORMAL_SNIPER,
    HEAVEN_HALF_HALF,
    HEAVEN_ALL_HIGH,
    HEAVEN_CLUSTER
};

const std::string PATTERN_NAMES[] = {
    "HELL_DEAD_SILENCE",
    "HELL_SLAUGHTER",
    "HELL_ONE_TRAP",
    "NORMAL_ACCIDENT",
    "NORMAL_PLAYABLE",
    "NORMAL_SNIPER",
    "HEAVEN_HALF_HALF",
    "HEAVEN_ALL_HIGH",
    "HEAVEN_CLUSTER"
};

// --- Data Structures ---
struct SimulationResult {
    int best_machine_bonus_prob_idx; // 0-49
    int worst_machine_games_idx;     // 0-19
    int island_avg_games_idx;        // 0-19
};

// Flat generic bin key for internal storage
struct BinData {
    uint32_t counts[NUM_PATTERNS];
};

// Grid dimensions
// Prob: 1/1 to 1/N? No, usually probability checks like 1/100 to 1/300. 
// Let's define the range for Best Machine Bonus Prob:
// 1/100 (Amazing) to 1/300 (Bad). 
// Wait, setting 1 is 1/170 (combined). 
// Range: 1/80 (Super Lucky) to 1/250 (Bad).
// Let's settle on: Probability denominator.
// Min Denom: 80, Max Denom: 300.
constexpr double MIN_DENOM = 80.0;
constexpr double MAX_DENOM = 300.0;
constexpr double DENOM_STEP = (MAX_DENOM - MIN_DENOM) / NUM_BINS_PROB;

// Games
constexpr int MAX_GAMES = 10000;
constexpr int GAME_STEP = MAX_GAMES / NUM_BINS_WORST; // 500

// --- Functions ---
void run_simulation_batch(int num_sims, int pattern_id, std::vector<BinData>& local_grid, uint64_t seed);
void save_to_json(const std::vector<BinData>& grid, const std::string& filename);
