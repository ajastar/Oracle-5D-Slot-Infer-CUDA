#include "matrix.h"
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  // defaults
  uint64_t trials = 300000000000ULL;
  int threads = 20;
  std::string outfile = "matrix_db_5d.bin";
  MachineType mtype = IM_JUGGLER;

  // args: [trials] [threads] [outfile] [machine_key]
  if (argc > 1)
    trials = std::stoull(argv[1]);
  if (argc > 2)
    threads = std::stoi(argv[2]);
  if (argc > 3)
    outfile = argv[3];
  if (argc > 4) {
    std::string s = argv[4];
    if (s == "my")
      mtype = MY_JUGGLER;
    else if (s == "gogo")
      mtype = GOGO_JUGGLER;
    else if (s == "happy")
      mtype = HAPPY_JUGGLER;
    else if (s == "funky")
      mtype = FUNKY_JUGGLER;
    else if (s == "ultra")
      mtype = ULTRA_MIRACLE;
    else if (s == "mr")
      mtype = MR_JUGGLER;
    else
      mtype = IM_JUGGLER;
  }

  run_oracle_simulation(trials, threads, outfile, mtype);
  return 0;
}
