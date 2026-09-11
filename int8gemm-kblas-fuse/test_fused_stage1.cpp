#include "tests/guarded_matrix.hpp"
#include "tests/reference.hpp"

#include <algorithm>
#include <cfenv>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <sys/prctl.h>
#include <vector>

#include "int8_gemm.hpp"

// Test-only control of the existing drivers; production defaults are unchanged.
extern int nthreadsM;
extern int nthreadsN;

namespace {

constexpr int kK = fusion_test::kFixedK;
constexpr int kMinBusinessDimension = 2048;
constexpr int kMaxBusinessDimension = 8192;

struct Options {
    int m = 2048;
    int n = 2048;
    int threads = 32;
    int repeat = 1;
    uint32_t seed = 42;
    std::vector<unsigned> modes = {0, 1, 19};
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
    return {unique.begin(), unique.end()};
}

Options parse_options(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            std::cout
                << "Stage 1 fused-store comparison; K=2048 only.\n"
                << "Usage: " << argv[0]
                << " [--m 2048|4096|6144|8192] [--n 2048|4096|6144|8192]\n"
                << "       [--threads 1|32] [--modes 0,1,19|all]"
                << " [--repeat 1..100] [--seed N]\n"
                << "The original path is the C32/C8 shadow reference.  The fused path\n"
                << "must match it exactly, and both INT8 outputs are checked against\n"
                << "the independent centered-remainder oracle.\n";
            std::exit(0);
        }
        if (i + 1 == argc) throw std::invalid_argument("missing value for " + arg);
        const std::string value = argv[++i];
        if (arg == "--m") options.m = parse_number(value, kMinBusinessDimension, kMaxBusinessDimension);
        else if (arg == "--n") options.n = parse_number(value, kMinBusinessDimension, kMaxBusinessDimension);
        else if (arg == "--threads") options.threads = parse_number(value, 1, 32);
        else if (arg == "--modes") options.modes = parse_modes(value);
        else if (arg == "--repeat") options.repeat = parse_number(value, 1, 100);
        else if (arg == "--seed") options.seed = parse_number(value, 0, 2147483647);
        else throw std::invalid_argument("unknown option: " + arg);
    }
    if (options.m % kMinBusinessDimension != 0 ||
        options.n % kMinBusinessDimension != 0) {
        throw std::invalid_argument("stage 1 accepts only M/N multiples of 2048");
    }
    if (options.threads != 1 && options.threads != 32) {
        throw std::invalid_argument("stage 1 tests 1x1 and 32x1 thread layouts only");
    }
    return options;
}

bool worker_environment_ok() {
    if (std::fesetround(FE_TONEAREST) != 0 || std::fegetround() != FE_TONEAREST) return false;
#if defined(__aarch64__)
    constexpr int kPrSveGetVl = 51;
    constexpr int kPrSmeGetVl = 64;
    constexpr long kVlLengthMask = 0xffff;
    const long vl = prctl(kPrSveGetVl, 0UL, 0UL, 0UL, 0UL);
    const long svl = prctl(kPrSmeGetVl, 0UL, 0UL, 0UL, 0UL);
    return vl >= 0 && svl >= 0 && (vl & kVlLengthMask) == 64 &&
           (svl & kVlLengthMask) == 64;
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
        throw std::runtime_error(
            "target preflight failed: require AArch64, SVE VL=64B, SME SVL=64B, "
            "FE_TONEAREST and the full requested OpenMP team");
    }
}

struct FreeAligned {
    void operator()(int8_t* pointer) const { std::free(pointer); }
};
using AlignedBuffer = std::unique_ptr<int8_t, FreeAligned>;

AlignedBuffer allocate_scratch(size_t bytes) {
    void* pointer = nullptr;
    if (posix_memalign(&pointer, 4096, bytes) != 0) throw std::bad_alloc();
    std::memset(pointer, 0, bytes);
    return AlignedBuffer(static_cast<int8_t*>(pointer));
}

void fill_inputs(const Options& options, std::vector<int8_t>& a, int lda,
                 std::vector<int8_t>& b, int ldb) {
    std::mt19937 rng(options.seed);
    for (int l = 0; l < kK; ++l) {
        for (int i = 0; i < options.m; ++i) {
            a[static_cast<size_t>(l) * lda + i] =
                static_cast<int8_t>(static_cast<int>(rng() % 256) - 128);
        }
    }
    for (int j = 0; j < options.n; ++j) {
        for (int l = 0; l < kK; ++l) {
            b[static_cast<size_t>(j) * ldb + l] =
                static_cast<int8_t>(static_cast<int>(rng() % 256) - 128);
        }
    }
}

template <typename T>
void require_same_matrix(const fusion_test::GuardedMatrix<T>& original,
                         const fusion_test::GuardedMatrix<T>& fused,
                         int rows, int cols, int original_ld, int fused_ld,
                         const std::string& name) {
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            const T lhs = original.data()[static_cast<size_t>(j) * original_ld + i];
            const T rhs = fused.data()[static_cast<size_t>(j) * fused_ld + i];
            if (lhs != rhs) {
                throw std::runtime_error(name + " mismatch row=" + std::to_string(i) +
                                         " col=" + std::to_string(j) +
                                         " original=" + std::to_string(static_cast<int64_t>(lhs)) +
                                         " fused=" + std::to_string(static_cast<int64_t>(rhs)));
            }
        }
    }
}

void verify_inverse_output(const fusion_test::GuardedMatrix<int32_t>& c32,
                           const fusion_test::GuardedMatrix<int8_t>& c8,
                           int rows, int cols, int ldc32, int ldc8,
                           unsigned mode, const std::string& which) {
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            const int32_t value = c32.data()[static_cast<size_t>(j) * ldc32 + i];
            const int expected = fusion_test::inverse_integer(value, mode);
            const int legacy = fusion_test::inverse_fp64(value, mode);
            const int actual = c8.data()[static_cast<size_t>(j) * ldc8 + i];
            if (legacy != expected || actual != expected) {
                throw std::runtime_error(
                    which + " inverse mismatch mode=" + std::to_string(mode) +
                    " row=" + std::to_string(i) + " col=" + std::to_string(j) +
                    " C32=" + std::to_string(value) +
                    " C8=" + std::to_string(actual) +
                    " integer8=" + std::to_string(expected) +
                    " fp64_legacy8=" + std::to_string(legacy));
            }
        }
    }
}

size_t run(const Options& options, int8_t* sa, int8_t* sb) {
    // Deliberately use distinct padded strides: phase 1 must not accidentally
    // advance the INT8 destination using the INT32 leading dimension.
    const int lda = options.m + 7;
    const int ldb = kK + 11;
    const int ldc32 = options.m + 16;
    const int ldc8 = options.m + 23;
    std::cout << "RUN STAGE1 M=" << options.m << " N=" << options.n
              << " K=" << kK << " threads=" << options.threads
              << " lda=" << lda << " ldb=" << ldb << " ldc32=" << ldc32
              << " ldc8=" << ldc8 << " modes=";
    for (size_t i = 0; i < options.modes.size(); ++i) {
        if (i != 0) std::cout << ',';
        std::cout << options.modes[i];
    }
    std::cout << " repeat=" << options.repeat << " seed=" << options.seed << '\n';

    check_team(options.threads);
    nthreadsM = options.threads;
    nthreadsN = 1;

    std::vector<int8_t> a(static_cast<size_t>(lda) * kK, 93);
    std::vector<int8_t> b(static_cast<size_t>(ldb) * options.n, 93);
    fill_inputs(options, a, lda, b, ldb);
    const auto original_a = a;
    const auto original_b = b;

    fusion_test::GuardedMatrix<int32_t> original_c32(options.m, options.n, ldc32, 0x5a6b7c1d);
    fusion_test::GuardedMatrix<int32_t> fused_c32(options.m, options.n, ldc32, 0x31415926);
    fusion_test::GuardedMatrix<int8_t> original_c8(options.m, options.n, ldc8, 85);
    fusion_test::GuardedMatrix<int8_t> fused_c8(options.m, options.n, ldc8, -86);
    const int32_t offset = 0;
    size_t passed = 0;

    for (unsigned mode : options.modes) {
        for (int repeat = 0; repeat < options.repeat; ++repeat) {
            original_c32.reset(repeat % 2 == 0 ? 0 : 0x12345678);
            original_c8.reset(repeat % 2 == 0 ? -91 : 91);
            fused_c32.reset(repeat % 2 == 0 ? 0 : 0x2468ace0);
            fused_c8.reset(repeat % 2 == 0 ? -73 : 73);

            cblas_gemm_s8s8s32(
                CblasColMajor, CblasNoTrans, CblasNoTrans, CblasFixOffset,
                options.m, options.n, kK, 1.0f, a.data(), lda, 0,
                b.data(), ldb, 0, 0.0f, original_c32.data(), ldc32, &offset,
                sa, sb, original_c8.data(), static_cast<size_t>(ldc8), mode);

            cblas_gemm_s8s8s32_fused(
                CblasColMajor, CblasNoTrans, CblasNoTrans, CblasFixOffset,
                options.m, options.n, kK, 1.0f, a.data(), lda, 0,
                b.data(), ldb, 0, 0.0f, fused_c32.data(), ldc32, &offset,
                sa, sb, fused_c8.data(), static_cast<size_t>(ldc8), mode);

            const std::string context = " mode=" + std::to_string(mode) +
                                        " repeat=" + std::to_string(repeat);
            original_c32.check_guards("original C32" + context);
            fused_c32.check_guards("fused C32" + context);
            original_c8.check_guards("original C8" + context);
            fused_c8.check_guards("fused C8" + context);
            require_same_matrix(original_c32, fused_c32, options.m, options.n,
                                ldc32, ldc32, "C32" + context);
            require_same_matrix(original_c8, fused_c8, options.m, options.n,
                                ldc8, ldc8, "C8" + context);
            verify_inverse_output(original_c32, original_c8, options.m, options.n,
                                  ldc32, ldc8, mode, "original" + context);
            verify_inverse_output(fused_c32, fused_c8, options.m, options.n,
                                  ldc32, ldc8, mode, "fused" + context);
            if (a != original_a || b != original_b) {
                throw std::runtime_error("input modified" + context);
            }
            ++passed;
        }
    }
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        omp_set_dynamic(0);
        omp_set_max_active_levels(1);
        if (std::fesetround(FE_TONEAREST) != 0) {
            throw std::runtime_error("cannot set FE_TONEAREST");
        }
        // Current driver contract: P=128, Q=2048, R=8192, plus prefetch slack.
        auto sa = allocate_scratch(static_cast<size_t>(kK) * 128 * 32 + 65536);
        auto sb = allocate_scratch(static_cast<size_t>(kK) * 8192 + 65536);
        const size_t passed = run(options, sa.get(), sb.get());
        std::cout << "STAGE1 PASS: " << passed
                  << " original-vs-fused calls exact; C32 shadow, C8, guards "
                     "and inputs verified.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "STAGE1 FAIL: " << error.what() << '\n';
        return 1;
    }
}
