#include "tests/guarded_matrix.hpp"
#include "tests/reference.hpp"

#include <cfenv>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <sys/prctl.h>

#include "int8_gemm.hpp"

// Test-only control of the production driver thread layout.
extern int nthreadsM;
extern int nthreadsN;

namespace {
constexpr int kK = fusion_test::kFixedK;
constexpr int kDimensionBlock = 2048;
constexpr int kMaxDimension = 8192;

enum class Pattern { Random, Zero, Positive, Negative, Checker, Precision };
const char* pattern_name(Pattern pattern) {
    switch (pattern) {
        case Pattern::Random: return "random";
        case Pattern::Zero: return "zero";
        case Pattern::Positive: return "positive-extreme";
        case Pattern::Negative: return "negative-extreme";
        case Pattern::Checker: return "checker";
        case Pattern::Precision: return "fp32-sensitive";
    }
    throw std::logic_error("unknown pattern");
}
struct Case { int m, n, threads; Pattern pattern; bool padded; };
struct Options {
    std::string suite = "smoke";
    int m = 0, n = 0, threads = 32, repeat = 1;
    uint32_t seed = 42;
    bool custom_threads = false;
    std::vector<unsigned> modes = [] {
        std::vector<unsigned> result;
        for (unsigned mode = 0; mode <= fusion_test::kModuli.size(); ++mode) {
            result.push_back(mode);
        }
        return result;
    }();
};

int parse_number(const std::string& text, int lower, int upper) {
    size_t used = 0;
    const long long value = std::stoll(text, &used);
    if (used != text.size() || value < lower || value > upper) {
        throw std::invalid_argument("numeric argument out of range: " + text);
    }
    return static_cast<int>(value);
}

std::vector<unsigned> parse_modes(const std::string& text) {
    if (text == "all") {
        std::vector<unsigned> result;
        for (unsigned mode = 0; mode <= fusion_test::kModuli.size(); ++mode) {
            result.push_back(mode);
        }
        return result;
    }

    std::set<unsigned> unique;
    std::stringstream stream(text);
    std::string item;
    while (std::getline(stream, item, ',')) {
        if (item.empty()) throw std::invalid_argument("empty item in --modes");
        unique.insert(static_cast<unsigned>(parse_number(
            item, 0, static_cast<int>(fusion_test::kModuli.size()))));
    }
    if (unique.empty()) throw std::invalid_argument("--modes cannot be empty");
    return std::vector<unsigned>(unique.begin(), unique.end());
}

std::string modes_name(const std::vector<unsigned>& modes) {
    std::ostringstream result;
    for (size_t i = 0; i < modes.size(); ++i) {
        if (i != 0) result << ',';
        result << modes[i];
    }
    return result.str();
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout << "Production fused GEMM, K=2048 only; all num_moduli=0..19.\n"
                      << "Usage: " << argv[0] << " [--suite smoke|full] [--repeat 1..100] [--seed N]\n"
                      << "       " << argv[0] << " --m M --n N [--threads 1|32] [--modes all|0,1,19]\n"
                      << "Custom dimensions: M/N must be positive multiples of 2048 (max 8192).\n"
                      << "At 32 threads, N must be divisible by 32.\n"
                      << "Requires target SME INT8 hardware with SVE VL=SME SVL=64 bytes.\n";
            std::exit(0);
        }
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--suite") options.suite = value;
        else if (arg == "--m") options.m = parse_number(value, 1, kMaxDimension);
        else if (arg == "--n") options.n = parse_number(value, 1, kMaxDimension);
        else if (arg == "--threads") {
            options.threads = parse_number(value, 1, 32);
            options.custom_threads = true;
        }
        else if (arg == "--modes") options.modes = parse_modes(value);
        else if (arg == "--repeat") options.repeat = parse_number(value, 1, 100);
        else if (arg == "--seed") options.seed = parse_number(value, 0, 2147483647);
        else throw std::invalid_argument("unknown option: " + arg);
    }
    if (options.suite != "smoke" && options.suite != "full") {
        throw std::invalid_argument("suite must be smoke or full");
    }
    if ((options.m == 0) != (options.n == 0)) {
        throw std::invalid_argument("--m and --n must be provided together");
    }
    if (options.threads != 1 && options.threads != 32) {
        throw std::invalid_argument("production tests thread layouts 1x1 and 32x1 only");
    }
    if (options.m == 0 && options.custom_threads) {
        throw std::invalid_argument("--threads applies to a custom --m/--n case only");
    }
    if (options.m != 0 && options.threads == 32 && options.n % 32 != 0) {
        throw std::invalid_argument("32-thread production path requires N divisible by 32; "
                                    "use --threads 1 to isolate an N tail");
    }
    if (options.m != 0 &&
        (options.m % kDimensionBlock != 0 || options.n % kDimensionBlock != 0)) {
        throw std::invalid_argument("production contract requires M and N to be multiples of 2048");
    }
    return options;
}

std::vector<Case> make_cases(const Options& options) {
    if (options.m != 0) {
        return {{options.m, options.n, options.threads, Pattern::Random, true}};
    }
    std::vector<Case> cases = {
        {kDimensionBlock, kDimensionBlock, 1, Pattern::Random, false},
        {kDimensionBlock, kDimensionBlock, 32, Pattern::Random, true}
    };
    if (options.suite == "full") {
        cases.push_back({2 * kDimensionBlock, 2 * kDimensionBlock, 32,
                         Pattern::Precision, true});
        cases.push_back({4 * kDimensionBlock, 4 * kDimensionBlock, 32,
                         Pattern::Random, true});
    }
    return cases;
}

bool worker_environment_ok() {
    if (std::fesetround(FE_TONEAREST) != 0 || std::fegetround() != FE_TONEAREST) return false;
#if defined(__aarch64__)
    // Linux UAPI request numbers; keep usable with older target sysroot headers.
    constexpr int kPrSveGetVl = 51;
    constexpr int kPrSmeGetVl = 64;
    constexpr long kVlLengthMask = 0xffff;
    const long vl = prctl(kPrSveGetVl, 0UL, 0UL, 0UL, 0UL);
    const long svl = prctl(kPrSmeGetVl, 0UL, 0UL, 0UL, 0UL);
    return vl >= 0 && svl >= 0 && (vl & kVlLengthMask) == 64 && (svl & kVlLengthMask) == 64;
#else
    return false;
#endif
}

void check_team(int threads) {
    int bad = 0;
#pragma omp parallel num_threads(threads) reduction(+:bad)
    {
        if (omp_get_num_threads() != threads || !worker_environment_ok()) ++bad;
    }
    if (bad != 0) {
        throw std::runtime_error("target preflight failed: require AArch64, SVE VL=64B, "
                                 "SME SVL=64B, FE_TONEAREST and the full requested OpenMP team; "
                                 "check OMP_THREAD_LIMIT and target vector-length settings");
    }
}

struct FreeAligned { void operator()(int8_t* p) const { std::free(p); } };
using AlignedBuffer = std::unique_ptr<int8_t, FreeAligned>;
AlignedBuffer allocate_scratch(size_t bytes) {
    void* pointer = nullptr;
    if (posix_memalign(&pointer, 4096, bytes) != 0) throw std::bad_alloc();
    std::memset(pointer, 0, bytes);
    return AlignedBuffer(static_cast<int8_t*>(pointer));
}

void fill_inputs(const Case& test, uint32_t seed, std::vector<int8_t>& a, int lda,
                 std::vector<int8_t>& b, int ldb) {
    std::mt19937 rng(seed);
    for (int l = 0; l < kK; ++l) {
        for (int i = 0; i < test.m; ++i) {
            int value = 0;
            switch (test.pattern) {
                case Pattern::Random: value = static_cast<int>(rng() % 256) - 128; break;
                case Pattern::Zero: value = 0; break;
                case Pattern::Positive: value = 127; break;
                case Pattern::Negative: value = -128; break;
                case Pattern::Checker: value = ((i + l) & 1) ? -128 : 127; break;
                case Pattern::Precision: value = (i & 1 ? -1 : 1) * (l == kK - 1 ? 126 : 127); break;
            }
            a[static_cast<size_t>(l) * lda + i] = static_cast<int8_t>(value);
        }
    }
    for (int j = 0; j < test.n; ++j) {
        for (int l = 0; l < kK; ++l) {
            int value = 0;
            switch (test.pattern) {
                case Pattern::Random: value = static_cast<int>(rng() % 256) - 128; break;
                case Pattern::Zero: value = 0; break;
                case Pattern::Positive: case Pattern::Negative: value = 127; break;
                case Pattern::Checker: value = ((j + l) & 1) ? -128 : 127; break;
                case Pattern::Precision: value = j & 1 ? -127 : 127; break;
            }
            b[static_cast<size_t>(j) * ldb + l] = static_cast<int8_t>(value);
        }
    }
}

struct ReferenceProbe {
    int row;
    int col;
    int32_t expected;
};

// The production contract starts at 2048x2048.  A full scalar M*N*K oracle is
// therefore not a practical target test (8192x8192x2048 would require more
// than 137 billion multiply-adds).  Keep an independent C32 oracle at a
// bounded set of coordinates that covers the beginnings and ends of every
// 2048 block, the 128-row/32-column kernel boundaries, and deterministic
// interior points.  The fused C8 relation is still checked for every element.
std::vector<std::pair<int, int>> reference_probe_coordinates(int m, int n) {
    std::set<int> rows;
    std::set<int> cols;
    const int block_offsets[] = {0, 1, 31, 32, 63, 64, 127, 128, 129,
                                 kDimensionBlock - 2, kDimensionBlock - 1};
    for (int base = 0; base < m; base += kDimensionBlock) {
        for (int offset : block_offsets) {
            if (base + offset < m) rows.insert(base + offset);
        }
    }
    for (int base = 0; base < n; base += kDimensionBlock) {
        for (int offset : block_offsets) {
            if (base + offset < n) cols.insert(base + offset);
        }
    }

    // Add deterministic interior probes so a boundary-only implementation
    // cannot pass while producing a bad value in the middle of a tile.
    uint32_t state = 0x9e3779b9u ^ static_cast<uint32_t>(m * 17 + n);
    for (int sample = 0; sample < 128; ++sample) {
        state = state * 1664525u + 1013904223u;
        rows.insert(static_cast<int>(state % static_cast<uint32_t>(m)));
        state = state * 1664525u + 1013904223u;
        cols.insert(static_cast<int>(state % static_cast<uint32_t>(n)));
    }

    std::vector<std::pair<int, int>> coordinates;
    coordinates.reserve(rows.size() * cols.size());
    for (int row : rows) {
        for (int col : cols) coordinates.emplace_back(row, col);
    }
    return coordinates;
}

size_t run_case(const Case& test, const Options& options, int8_t* sa, int8_t* sb) {
    const int lda = test.m + (test.padded ? 7 : 0);
    const int ldb = kK + (test.padded ? 11 : 0);
    const int ldc = test.m + (test.padded ? 16 : 0);
    const int ldc8 = test.m + (test.padded ? 23 : 0);
    std::cout << "RUN M=" << test.m << " N=" << test.n << " K=" << kK
              << " threads=" << test.threads << " lda=" << lda << " ldb=" << ldb
              << " ldc32=" << ldc << " ldc8=" << ldc8
              << " pattern=" << pattern_name(test.pattern) << " seed=" << options.seed
              << " modes=" << modes_name(options.modes) << std::endl;
    check_team(test.threads);
    nthreadsM = test.threads;
    nthreadsN = 1;
    std::vector<int8_t> a(static_cast<size_t>(lda) * kK, 93);
    std::vector<int8_t> b(static_cast<size_t>(ldb) * test.n, 93);
    fill_inputs(test, options.seed, a, lda, b, ldb);
    const auto original_a = a, original_b = b;
    std::vector<ReferenceProbe> probes;
    for (const auto& coordinate : reference_probe_coordinates(test.m, test.n)) {
        probes.push_back({coordinate.first, coordinate.second,
                          fusion_test::gemm_reference_element(
                              a.data(), lda, b.data(), ldb,
                              coordinate.first, coordinate.second)});
    }
    if (test.pattern == Pattern::Precision &&
        probes.front().expected != 33032065) {
        throw std::logic_error("precision input no longer exercises a large odd INT32 result");
    }
    fusion_test::GuardedMatrix<int32_t> c32(test.m, test.n, ldc, 0x5a6b7c1d);
    fusion_test::GuardedMatrix<int8_t> c8(test.m, test.n, ldc8, 85);
    const int32_t offset = 0;
    size_t passed = 0;
    for (unsigned mode : options.modes) {
        for (int repeat = 0; repeat < options.repeat; ++repeat) {
            // beta=0 must overwrite C, independent of its previous contents.
            c32.reset(repeat % 2 == 0 ? 0 : 0x12345678);
            c8.reset(repeat % 2 == 0 ? -91 : 91);
            cblas_gemm_s8s8s32(CblasColMajor, CblasNoTrans, CblasNoTrans, CblasFixOffset,
                test.m, test.n, kK, 1.0f, a.data(), lda, 0, b.data(), ldb, 0, 0.0f,
                c32.data(), ldc, &offset, sa, sb, c8.data(), static_cast<size_t>(ldc8), mode);
            const std::string context = " mode=" + std::to_string(mode) +
                                        " repeat=" + std::to_string(repeat);
            c32.check_guards("C32" + context);
            c8.check_guards("C8" + context);
            for (const ReferenceProbe& probe : probes) {
                const int32_t actual32 =
                    c32.data()[static_cast<size_t>(probe.col) * ldc + probe.row];
                if (actual32 != probe.expected) {
                    throw std::runtime_error("C32 probe mismatch" + context +
                        " row=" + std::to_string(probe.row) + " col=" +
                        std::to_string(probe.col) + " C32=" +
                        std::to_string(actual32) + " expected32=" +
                        std::to_string(probe.expected));
                }
            }
            for (int j = 0; j < test.n; ++j) {
                for (int i = 0; i < test.m; ++i) {
                    const int32_t actual32 = c32.data()[static_cast<size_t>(j) * ldc + i];
                    const int expected8 = fusion_test::inverse_integer(actual32, mode);
                    const int legacy8 = fusion_test::inverse_fp64(actual32, mode);
                    const int actual8 = c8.data()[static_cast<size_t>(j) * ldc8 + i];
                    if (actual8 != expected8 || legacy8 != expected8) {
                        throw std::runtime_error("output mismatch" + context + " row=" +
                            std::to_string(i) + " col=" + std::to_string(j) + " C32=" +
                            std::to_string(actual32) +
                            " C8=" + std::to_string(actual8) + " integer8=" + std::to_string(expected8) +
                            " fp64_legacy8=" + std::to_string(legacy8));
                    }
                }
            }
            if (a != original_a || b != original_b) throw std::runtime_error("input modified" + context);
            ++passed;
        }
    }
    std::cout << "PASS " << passed
              << " calls; C8 inverse relation exhaustive, C32 independent probes, "
                 "output guards and inputs exact.\n";
    return passed;
}
}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        omp_set_dynamic(0);
        omp_set_max_active_levels(1);
        if (std::fesetround(FE_TONEAREST) != 0) throw std::runtime_error("cannot set FE_TONEAREST");
        // Use the current P=128, Q=2048, R=8192 driver contract, plus prefetch slack.
        // Ordinary aligned memory is sufficient for correctness; this is not an HBM benchmark.
        auto sa = allocate_scratch(static_cast<size_t>(kK) * 128 * 32 + 65536);
        auto sb = allocate_scratch(static_cast<size_t>(kK) * 8192 + 65536);
        size_t passed = 0;
        const auto cases = make_cases(options);
        for (const Case& test : cases) passed += run_case(test, options, sa.get(), sb.get());
        std::cout << "PRODUCTION PASS: " << passed << " calls across " << cases.size()
                  << " business-shape cases; K=2048, modes="
                  << modes_name(options.modes) << ".\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "PRODUCTION FAIL: " << e.what() << '\n';
        return 1;
    }
}
