#include <chrono>
#include <cfenv>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <hbwmalloc.h>
#include <numa.h>
#include <numaif.h>
#include <omp.h>
#include <sched.h>
#include <sys/mman.h>

#include "int8_gemm.hpp"

// The production driver exposes these variables for test-time thread control.
extern int nthreadsM;
extern int nthreadsN;

namespace {

constexpr int kFixedK = 2048;
constexpr int kDimensionBlock = 2048;
constexpr int kMaxDimension = 8192;
constexpr unsigned kModulusCount = 19;
constexpr double kPeakGflops = 1.55 * 128.0 * 32.0 * 16.0;

// Keep the same scratch sizing as the original int8/test_unigemm benchmark.
constexpr size_t kSaBytes = static_cast<size_t>(2048) * 128 * 2 * 32 * 2;
constexpr size_t kSbBytes = static_cast<size_t>(2048) * 8192 * 2 * 2;

int parse_int(const std::string& text, const char* name, int min_value, int max_value) {
    size_t used = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &used);
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + text);
    }
    if (used != text.size() || value < min_value || value > max_value) {
        throw std::invalid_argument(std::string(name) + " out of range: " + text);
    }
    return static_cast<int>(value);
}

std::vector<unsigned> parse_modes(const std::string& text) {
    if (text == "all") {
        std::vector<unsigned> modes;
        for (unsigned mode = 0; mode <= kModulusCount; ++mode) modes.push_back(mode);
        return modes;
    }

    std::vector<unsigned> modes;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) throw std::invalid_argument("empty mode in: " + text);
        const int mode = parse_int(item, "mode", 0, static_cast<int>(kModulusCount));
        modes.push_back(static_cast<unsigned>(mode));
    }
    if (modes.empty()) throw std::invalid_argument("modes cannot be empty");
    return modes;
}

std::string modes_to_string(const std::vector<unsigned>& modes) {
    std::ostringstream output;
    for (size_t i = 0; i < modes.size(); ++i) {
        if (i != 0) output << ',';
        output << modes[i];
    }
    return output.str();
}

void print_usage(const char* program) {
    std::cout
        << "Usage: " << program << " M N K [threads] [modes] [repeat] [warmup]\n"
        << "  M/N: positive multiples of 2048, maximum 8192\n"
        << "  K:    must be 2048\n"
        << "  threads: 1 or 32 (default 32)\n"
        << "  modes: 0..19, comma separated, or all (default 0,1,19)\n"
        << "  repeat: timed calls per mode (default 1)\n"
        << "  warmup: untimed calls per mode (default 0)\n"
        << "\nExamples:\n"
        << "  " << program << " 2048 2048 2048 32 0,1,19 3 1\n"
        << "  " << program << " 8192 8192 2048 32 0,1,19 3 1\n";
}

struct Options {
    int m;
    int n;
    int k;
    int threads = 32;
    std::vector<unsigned> modes = {0, 1, 19};
    int repeat = 1;
    int warmup = 0;
};

Options parse_options(int argc, char** argv) {
    if (argc < 4 || argc > 8) {
        print_usage(argv[0]);
        throw std::invalid_argument("expected 3 to 7 positional arguments");
    }

    Options options{
        parse_int(argv[1], "M", kDimensionBlock, kMaxDimension),
        parse_int(argv[2], "N", kDimensionBlock, kMaxDimension),
        parse_int(argv[3], "K", kFixedK, kFixedK),
    };
    if (argc >= 5) options.threads = parse_int(argv[4], "threads", 1, 32);
    if (argc >= 6) options.modes = parse_modes(argv[5]);
    if (argc >= 7) options.repeat = parse_int(argv[6], "repeat", 1, 1000);
    if (argc >= 8) options.warmup = parse_int(argv[7], "warmup", 0, 1000);

    if (options.m % kDimensionBlock != 0 || options.n % kDimensionBlock != 0) {
        throw std::invalid_argument("M and N must be multiples of 2048");
    }
    if (options.threads != 1 && options.threads != 32) {
        throw std::invalid_argument("threads must be 1 or 32");
    }
    if (options.threads == 32 && options.n % 32 != 0) {
        throw std::invalid_argument("N must be divisible by 32 for 32 threads");
    }
    return options;
}

struct HbmBuffer {
    void* data = nullptr;
    size_t bytes = 0;

    HbmBuffer() = default;
    HbmBuffer(void* pointer, size_t length) noexcept : data(pointer), bytes(length) {}
    HbmBuffer(const HbmBuffer&) = delete;
    HbmBuffer& operator=(const HbmBuffer&) = delete;

    HbmBuffer(HbmBuffer&& other) noexcept : data(other.data), bytes(other.bytes) {
        other.data = nullptr;
        other.bytes = 0;
    }
    HbmBuffer& operator=(HbmBuffer&& other) noexcept {
        if (this != &other) {
            release();
            data = other.data;
            bytes = other.bytes;
            other.data = nullptr;
            other.bytes = 0;
        }
        return *this;
    }

    ~HbmBuffer() { release(); }

    void release() {
        if (data != nullptr) {
            if (munmap(data, bytes) != 0) {
                std::perror("munmap");
            }
            data = nullptr;
            bytes = 0;
        }
    }
};

HbmBuffer allocate_hbm(size_t bytes, const char* name) {
    void* mapping = nullptr;
    mapping = mmap(nullptr, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
    if (mapping == MAP_FAILED) {
        std::fprintf(stderr, "mmap HBM allocation failed for %s (%zu bytes)\n", name, bytes);
        std::exit(EXIT_FAILURE);
    }

    const int cpu = sched_getcpu();
    const unsigned long node = numa_node_of_cpu(cpu);
    const int configured_nodes = numa_num_configured_nodes();
    const unsigned long hbm_node = node + static_cast<unsigned long>(configured_nodes / 2);

    nodemask_t nodemask{};
    struct bitmask bitmask = {NUMA_NUM_NODES, nodemask.n};
    numa_bitmask_clearall(&bitmask);
    numa_bitmask_setbit(&bitmask, hbm_node);

    if (mbind(mapping, bytes, MPOL_BIND, nodemask.n, NUMA_NUM_NODES, 0) != 0) {
        std::fprintf(stderr, "mbind HBM allocation failed for %s\n", name);
        munmap(mapping, bytes);
        std::exit(EXIT_FAILURE);
    }
    if (hbw_verify_memory_region(mapping, bytes, HBW_TOUCH_PAGES) != 0) {
        std::fprintf(stderr, "%s is not in HBM\n", name);
        munmap(mapping, bytes);
        std::exit(EXIT_FAILURE);
    }

    return HbmBuffer{mapping, bytes};
}

// Deterministic input initialization.  It is intentionally outside the timed
// region, while touching every valid element to establish the HBM placement.
uint32_t next_random(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state;
}

void fill_a(int8_t* a, int m, int k, int lda, uint32_t seed) {
    uint32_t state = seed;
    for (int col = 0; col < k; ++col) {
        for (int row = 0; row < m; ++row) {
            a[static_cast<size_t>(col) * lda + row] =
                static_cast<int8_t>(static_cast<int>((next_random(state) >> 24) & 0xffu) - 128);
        }
    }
}

void fill_b(int8_t* b, int k, int n, int ldb, uint32_t seed) {
    uint32_t state = seed;
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < k; ++row) {
            b[static_cast<size_t>(col) * ldb + row] =
                static_cast<int8_t>(static_cast<int>((next_random(state) >> 24) & 0xffu) - 128);
        }
    }
}

// A small checksum is printed after timing.  It makes an accidental no-op or
// wrong output pointer visible without adding a full C8 reference computation.
uint64_t checksum_c8(const int8_t* c8, int m, int n, int ldc8) {
    uint64_t checksum = 0;
    const size_t count = static_cast<size_t>(ldc8) * n;
    // `m` is deliberately part of the interface: it documents that callers
    // checksum only the logical M rows, even though this benchmark uses ldc8=M.
    (void)m;
    for (size_t index = 0; index < count; index += 4096) {
        checksum = checksum * 1315423911u + static_cast<uint8_t>(c8[index]);
    }
    return checksum;
}

// A restricted OpenMP team would make the nominal 32-core peak denominator
// meaningless.  Fail before timing instead of silently reporting a misleading
// efficiency number.  This runs outside the measured region.
void check_omp_team(int requested_threads) {
    int bad = 0;
#pragma omp parallel num_threads(requested_threads) reduction(+:bad)
    {
        if (omp_get_num_threads() != requested_threads) ++bad;
    }
    if (bad != 0) {
        throw std::runtime_error("OpenMP did not create the requested team; "
                                 "check OMP_THREAD_LIMIT / affinity settings");
    }
}

double run_mode(const Options& options, unsigned mode,
                int8_t* a, int8_t* b, int8_t* c8,
                int8_t* sa, int8_t* sb,
                int lda, int ldb, int ldc8) {
    nthreadsM = options.threads;
    nthreadsN = 1;

    for (int i = 0; i < options.warmup; ++i) {
        cblas_gemm_s8s8s8(CblasColMajor, CblasNoTrans, CblasNoTrans,
                          options.m, options.n, options.k, 1.0f,
                          a, lda, 0, b, ldb, 0, 0.0f,
                          sa, sb, c8, static_cast<size_t>(ldc8), mode);
    }

    double total_seconds = 0.0;
    for (int i = 0; i < options.repeat; ++i) {
        // Do not include output initialization in the GEMM timing, matching
        // int8/test_unigemm.cpp.  The fused kernel overwrites every valid C8
        // element on each call.
        std::memset(c8, 0, static_cast<size_t>(ldc8) * options.n);
        const auto start = std::chrono::high_resolution_clock::now();
        cblas_gemm_s8s8s8(CblasColMajor, CblasNoTrans, CblasNoTrans,
                          options.m, options.n, options.k, 1.0f,
                          a, lda, 0, b, ldb, 0, 0.0f,
                          sa, sb, c8, static_cast<size_t>(ldc8), mode);
        const auto end = std::chrono::high_resolution_clock::now();
        const std::chrono::duration<double> elapsed = end - start;
        total_seconds += elapsed.count();
    }

    const double average_seconds = total_seconds / options.repeat;
    const double gflops = (2.0 * options.m * options.n * options.k / 1.0e9) /
                          average_seconds;
    // This is intentionally the same efficiency definition as test_unigemm:
    // (2*M*N*K / time) / (1.55 * 128 * 32 * 16).
    const double efficiency = gflops / kPeakGflops * 100.0;

    std::cout << "M=" << options.m
              << " N=" << options.n
              << " K=" << options.k
              << " threads=" << options.threads
              << " mode=" << mode
              << " ldc8=" << ldc8
              << " fused_time: " << std::fixed << std::setprecision(6)
              << average_seconds << " seconds"
              << " performance: " << gflops << " GFLOPs"
              << " effi : " << efficiency << "%"
              << " checksum=" << checksum_c8(c8, options.m, options.n, ldc8)
              << std::endl;
    return average_seconds;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        omp_set_dynamic(0);
        omp_set_max_active_levels(1);
        if (std::fesetround(FE_TONEAREST) != 0 || std::fegetround() != FE_TONEAREST) {
            throw std::runtime_error("cannot set FE_TONEAREST");
        }
        check_omp_team(options.threads);

        const int lda = options.m;
        const int ldb = options.k;
        const int ldc8 = options.m;
        const size_t a_bytes = static_cast<size_t>(lda) * options.k;
        const size_t b_bytes = static_cast<size_t>(ldb) * options.n;
        const size_t c8_bytes = static_cast<size_t>(ldc8) * options.n;

        std::cout << "=======================================================\n"
                  << "INT8 GEMM fused inverse-scaling performance test\n"
                  << "=======================================================\n"
                  << "M=" << options.m << " N=" << options.n << " K=" << options.k
                  << " threads=" << options.threads
                  << " modes=" << modes_to_string(options.modes)
                  << " repeat=" << options.repeat
                  << " warmup=" << options.warmup << "\n"
                  << "A=" << a_bytes << " B=" << b_bytes << " C8=" << c8_bytes
                  << " scratchA=" << kSaBytes << " scratchB=" << kSbBytes << "\n";

        HbmBuffer a_buffer = allocate_hbm(a_bytes, "A");
        HbmBuffer b_buffer = allocate_hbm(b_bytes, "B");
        HbmBuffer c8_buffer = allocate_hbm(c8_bytes, "C8");
        HbmBuffer sa_buffer = allocate_hbm(kSaBytes, "sa");
        HbmBuffer sb_buffer = allocate_hbm(kSbBytes, "sb");

        auto* a = static_cast<int8_t*>(a_buffer.data);
        auto* b = static_cast<int8_t*>(b_buffer.data);
        auto* c8 = static_cast<int8_t*>(c8_buffer.data);
        auto* sa = static_cast<int8_t*>(sa_buffer.data);
        auto* sb = static_cast<int8_t*>(sb_buffer.data);

        fill_a(a, options.m, options.k, lda, 42);
        fill_b(b, options.k, options.n, ldb, 123);
        std::memset(c8, 0, c8_bytes);
        std::memset(sa, 0, kSaBytes);
        std::memset(sb, 0, kSbBytes);

        double total_seconds = 0.0;
        for (unsigned mode : options.modes) {
            total_seconds += run_mode(options, mode, a, b, c8, sa, sb,
                                      lda, ldb, ldc8);
        }

        if (options.modes.size() > 1) {
            const double average_seconds = total_seconds / options.modes.size();
            const double gflops = (2.0 * options.m * options.n * options.k / 1.0e9) /
                                  average_seconds;
            const double efficiency = gflops / kPeakGflops * 100.0;
            std::cout << "average fused time = " << std::fixed << std::setprecision(6)
                      << average_seconds << ", average fused effi = "
                      << efficiency << "\n";
        }

        std::cout << "=======================================================\n";
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "ERROR: %s\n", error.what());
        return 2;
    }
}
