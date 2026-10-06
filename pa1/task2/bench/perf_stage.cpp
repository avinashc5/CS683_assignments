// perf_stage.cpp  UNGRADED: measure hardware counters for ONE kernel call region.
//
// Why this exists
// ---------------
// The project harness (src/main.cpp) always runs matmul_naive alongside the stage under
// test -- it needs naive both as the correctness reference and as the speedup baseline.
// So `perf stat -- ./bin/matmul simd 1024 1024 1024` counts 6 naive calls + 5 simd calls,
// and naive (6.5e9 instructions/call) drowns simd (2.3e8/call): every stage comes out
// within a few percent of every other. src/main.cpp must not be modified, so the
// measurement lives here instead.
//
// Counting the region, not the process
// ------------------------------------
// This opens the counters directly with perf_event_open(2) and brackets ONLY the kernel
// call(s) with IOC_ENABLE / IOC_DISABLE. Everything else -- process startup, allocation,
// the mt19937 fill, the checksum -- happens outside the enabled window and is never
// counted. That matters: the fill alone is ~1.2e8 instructions, which would inflate a
// 2.3e8-instruction kernel by ~50% if the whole process were counted.
//
// Usage
//   ./bin/perf_stage <stage> [M N K] [seed] [reps] [flush]
//     stage : naive | simd | prefetch | optimized
//     reps  : kernel calls inside the counted region (default 1). Totals and per-call
//             values are both printed.
//     flush : 1 (default) streams a buffer larger than L3 before the region, so the
//             first call starts with cold caches instead of reading A/B straight out of
//             L2/L3 where the fill left them. Use 0 to measure the warm case.
//
// Counters are user-mode only (exclude_kernel), so page-fault and other kernel-side work
// is excluded even if it lands inside the region.
//
// Needs kernel.perf_event_paranoid <= 1.  Nothing here is part of the graded submission.
#include <asm/unistd.h>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "matmul.h"
#include "utils.h"
using namespace pa1;

namespace {

constexpr int kMaxEvents = 5;

struct EventSpec {
    const char* name;
    std::uint32_t type;
    std::uint64_t config;
};

// PERF_TYPE_HW_CACHE config is (cache_id) | (op_id << 8) | (result_id << 16).
constexpr std::uint64_t hw_cache(std::uint64_t cache, std::uint64_t op, std::uint64_t res) {
    return cache | (op << 8) | (res << 16);
}

const EventSpec kEvents[kMaxEvents] = {
    {"instructions", PERF_TYPE_HARDWARE, PERF_COUNT_HW_INSTRUCTIONS},
    {"cycles",       PERF_TYPE_HARDWARE, PERF_COUNT_HW_CPU_CYCLES},
    {"L1D-miss",     PERF_TYPE_HW_CACHE,
     hw_cache(PERF_COUNT_HW_CACHE_L1D, PERF_COUNT_HW_CACHE_OP_READ,
              PERF_COUNT_HW_CACHE_RESULT_MISS)},
    {"LLC-miss",     PERF_TYPE_HW_CACHE,
     hw_cache(PERF_COUNT_HW_CACHE_LL, PERF_COUNT_HW_CACHE_OP_READ,
              PERF_COUNT_HW_CACHE_RESULT_MISS)},
    // l2_rqsts.miss on Ice Lake / Tiger Lake: event 0x24, umask 0x3f -> raw 0x3f24.
    // Cross-check with: perf stat -e l2_rqsts.miss,r3f24 -- <cmd>   (both columns equal).
    {"L2-miss",      PERF_TYPE_RAW, 0x3f24},
};

int perf_event_open(perf_event_attr* attr, pid_t pid, int cpu, int group_fd,
                    unsigned long flags) {
    return static_cast<int>(syscall(__NR_perf_event_open, attr, pid, cpu, group_fd, flags));
}

// Read format for a counter group with scaling info.
struct GroupRead {
    std::uint64_t nr;
    std::uint64_t time_enabled;
    std::uint64_t time_running;
    struct { std::uint64_t value; std::uint64_t id; } vals[kMaxEvents];
};

class RegionCounters {
public:
    bool open_all() {
        for (int i = 0; i < kMaxEvents; ++i) {
            perf_event_attr pe;
            std::memset(&pe, 0, sizeof pe);
            pe.type = kEvents[i].type;
            pe.size = sizeof pe;
            pe.config = kEvents[i].config;
            pe.disabled = (i == 0);          // only the leader starts disabled
            pe.exclude_kernel = 1;
            pe.exclude_hv = 1;
            if (i == 0) {
                pe.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_ID |
                                 PERF_FORMAT_TOTAL_TIME_ENABLED |
                                 PERF_FORMAT_TOTAL_TIME_RUNNING;
            }
            fds_[i] = perf_event_open(&pe, 0 /*self*/, -1 /*any cpu*/,
                                      (i == 0) ? -1 : fds_[0], 0);
            if (fds_[i] == -1) {
                std::fprintf(stderr,
                             "perf_event_open failed for '%s' (%s).\n"
                             "  Try: sudo sysctl kernel.perf_event_paranoid=1\n",
                             kEvents[i].name, std::strerror(errno));
                return false;
            }
            ids_[i] = 0;
            ioctl(fds_[i], PERF_EVENT_IOC_ID, &ids_[i]);
        }
        return true;
    }

    void start() {
        ioctl(fds_[0], PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
        ioctl(fds_[0], PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
    }
    void stop() { ioctl(fds_[0], PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP); }

    // Returns false if the read failed. Values are scaled if the PMU multiplexed.
    bool read_scaled(double out[kMaxEvents], double* scale_factor) {
        GroupRead gr;
        std::memset(&gr, 0, sizeof gr);
        if (read(fds_[0], &gr, sizeof gr) <= 0) return false;
        const double scale = (gr.time_running > 0)
                                 ? static_cast<double>(gr.time_enabled) / gr.time_running
                                 : 1.0;
        *scale_factor = scale;
        for (int i = 0; i < kMaxEvents; ++i) {
            out[i] = 0.0;
            for (std::uint64_t v = 0; v < gr.nr; ++v)
                if (gr.vals[v].id == ids_[i]) out[i] = gr.vals[v].value * scale;
        }
        return true;
    }

    void close_all() {
        for (int i = 0; i < kMaxEvents; ++i)
            if (fds_[i] != -1) close(fds_[i]);
    }

private:
    int fds_[kMaxEvents] = {-1, -1, -1, -1, -1};
    std::uint64_t ids_[kMaxEvents] = {0, 0, 0, 0, 0};
};

// Stream through a buffer larger than L3 (12 MiB here) to evict A/B/C before measuring.
void flush_caches() {
    const std::size_t n = 64u << 20;  // 64 MiB
    volatile char* buf = static_cast<volatile char*>(std::malloc(n));
    if (!buf) return;
    for (std::size_t i = 0; i < n; i += 64) buf[i] = static_cast<char>(i);
    std::free(const_cast<char*>(buf));
}

}  // namespace

int main(int argc, char** argv) {
    const char* stage = (argc > 1) ? argv[1] : "optimized";
    int M = (argc > 4) ? std::atoi(argv[2]) : 1024;
    int N = (argc > 4) ? std::atoi(argv[3]) : 1024;
    int K = (argc > 4) ? std::atoi(argv[4]) : 1024;
    unsigned seed = (argc > 5) ? static_cast<unsigned>(std::atoi(argv[5])) : 1234u;
    int reps = (argc > 6) ? std::atoi(argv[6]) : 1;
    int flush = (argc > 7) ? std::atoi(argv[7]) : 1;
    if (reps < 1) reps = 1;

    MatMulFn fn = nullptr;
    if      (!std::strcmp(stage, "naive"))     fn = matmul_naive;
    else if (!std::strcmp(stage, "simd"))      fn = matmul_simd;
    else if (!std::strcmp(stage, "prefetch"))  fn = matmul_prefetch;
    else if (!std::strcmp(stage, "optimized")) fn = matmul_optimized;
    else { std::fprintf(stderr, "unknown stage '%s'\n", stage); return 2; }

    float* A = alloc_floats(static_cast<std::size_t>(M) * K);
    float* B = alloc_floats(static_cast<std::size_t>(N) * K);
    float* C = alloc_floats(static_cast<std::size_t>(M) * N);
    fill_random(A, static_cast<std::size_t>(M) * K, seed);
    fill_random(B, static_cast<std::size_t>(N) * K, seed + 1);
    std::memset(C, 0, static_cast<std::size_t>(M) * N * sizeof(float));

    RegionCounters pc;
    if (!pc.open_all()) return 3;

    if (flush) flush_caches();

    // ---- counted region: nothing but the kernel ----
    pc.start();
    for (int r = 0; r < reps; ++r) fn(A, B, C, M, N, K, K, K, N);
    pc.stop();
    // ------------------------------------------------

    double v[kMaxEvents], scale = 1.0;
    const bool ok = pc.read_scaled(v, &scale);
    pc.close_all();
    if (!ok) { std::fprintf(stderr, "counter read failed\n"); return 4; }

    // Consume C so the kernel cannot be optimized away.
    double acc = 0.0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(M) * N; i += 4096) acc += C[i];

    std::printf("stage=%s M=%d N=%d K=%d reps=%d flush=%d%s\n", stage, M, N, K, reps,
                flush, (scale > 1.001) ? "  [multiplexed, values scaled]" : "");
    std::printf("%-14s %18s %18s\n", "event", "total", "per-call");
    for (int i = 0; i < kMaxEvents; ++i)
        std::printf("%-14s %18.0f %18.0f\n", kEvents[i].name, v[i], v[i] / reps);
    if (v[1] > 0) std::printf("%-14s %18.2f\n", "IPC", v[0] / v[1]);
    std::fprintf(stderr, "checksum=%.6f\n", acc);
    return 0;
}
