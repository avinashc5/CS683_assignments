#!/bin/bash
# Usage: scripts/build_task3.sh
# Builds every binary needed for Task 3 (names are the run_all.sh suffixes):
#   baseline    no prefetcher, LRU                       dynirrip      no prefetcher, dynamic IRRIP
#   spp         SPP, LRU                                 spp_dynnopf   SPP, dynamic IRRIP ignoring prefetches (Task 2B policy)
#   spp_dyn     SPP, prefetch-aware dynamic IRRIP (Task 3)
#   spp_dyn_pfr{1,2,3}  Task 3 with PF_INSERT_RRPV = 1/2/3 (sensitivity of the prefetch insertion position)
# The variants are generated from replacement/dyn_irrip.l2c_repl with sed and removed after the build.
cd "$(dirname "$0")/.."

./build_champsim.sh no  lru        baseline    || exit 1
./build_champsim.sh spp lru        spp         || exit 1
./build_champsim.sh no  dyn_irrip  dynirrip    || exit 1
./build_champsim.sh spp dyn_irrip  spp_dyn     || exit 1

# prefetch-unaware copy (PF_AWARE 0)
sed 's/#define PF_AWARE 1/#define PF_AWARE 0/' replacement/dyn_irrip.l2c_repl > replacement/dyn_irripnopf.l2c_repl
./build_champsim.sh spp dyn_irripnopf spp_dynnopf; rc=$?
rm -f replacement/dyn_irripnopf.l2c_repl
[ $rc -eq 0 ] || exit 1

# PF_INSERT_RRPV sweep
for r in 1 2 3; do
    sed "s/#define PF_INSERT_RRPV maxRRPV /#define PF_INSERT_RRPV $r /" replacement/dyn_irrip.l2c_repl > replacement/dyn_irrippfr$r.l2c_repl
    ./build_champsim.sh spp dyn_irrippfr$r spp_dyn_pfr$r; rc=$?
    rm -f replacement/dyn_irrippfr$r.l2c_repl
    [ $rc -eq 0 ] || exit 1
done
