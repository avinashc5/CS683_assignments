# bench/Makefile  UNGRADED benchmark harnesses for the Task 2 report plots.
#
# Nothing here is part of the graded submission. It builds:
#   bin/bench_simd_width   -- times naive vs SSE / AVX2 / AVX-512 variants (report plot 4)
#
# The size sweep and the two prefetch sweeps (plots 1-3) do NOT need anything here; they
# drive ../bin/matmul directly -- see run_size_sweep.sh / run_prefetch_degree.sh /
# run_prefetch_level.sh.
#
# Per the assignment, each SIMD-width variant is compiled with its own ISA flag while the
# scalar base optimisation level matches the pinned project flags (-O2 -fno-tree-vectorize).
#
#   make                 # build bin/bench_simd_width (with the AVX-512 TU)
#   make AVX512=0         # ... without the AVX-512 TU (toolchain lacks -mavx512f)
#   make clean

CXX      := g++
BASE     := -std=c++17 -O2 -fno-tree-vectorize -I../include -I. -Wall
SSEFLAGS := -msse4.2 -mfma
AVX2FLAGS:= -mavx2 -mfma
AVX512FLAGS := -mavx512f

BIN := bin
AVX512 ?= 1

OBJ := $(BIN)/matmul_naive.o $(BIN)/matmul_simd_sse.o $(BIN)/matmul_simd_avx2.o \
       $(BIN)/bench_simd_width.o

ifeq ($(AVX512),1)
  OBJ += $(BIN)/matmul_simd_avx512.o
  BASE += -DBENCH_HAVE_AVX512=1
else
  BASE += -DBENCH_HAVE_AVX512=0
endif

.PHONY: all clean
all: $(BIN)/bench_simd_width

$(BIN)/bench_simd_width: $(OBJ) | $(BIN)
	$(CXX) $(BASE) $^ -o $@

$(BIN)/matmul_naive.o: ../src/matmul_naive.cpp | $(BIN)
	$(CXX) $(BASE) -c $< -o $@

$(BIN)/matmul_simd_sse.o: matmul_simd_sse.cpp matmul_width.h | $(BIN)
	$(CXX) $(BASE) $(SSEFLAGS) -c $< -o $@

$(BIN)/matmul_simd_avx2.o: matmul_simd_avx2.cpp matmul_width.h | $(BIN)
	$(CXX) $(BASE) $(AVX2FLAGS) -c $< -o $@

$(BIN)/matmul_simd_avx512.o: matmul_simd_avx512.cpp matmul_width.h | $(BIN)
	$(CXX) $(BASE) $(AVX512FLAGS) -c $< -o $@

$(BIN)/bench_simd_width.o: bench_simd_width.cpp matmul_width.h | $(BIN)
	$(CXX) $(BASE) $(AVX2FLAGS) -c $< -o $@

$(BIN):
	mkdir -p $(BIN)

clean:
	rm -rf $(BIN)
