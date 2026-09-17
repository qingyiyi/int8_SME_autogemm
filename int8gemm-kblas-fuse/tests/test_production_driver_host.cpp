// Portable production-driver test.  It replaces the target-only packers and
// SME kernel with strict mocks, so it verifies that the original
// cblas_gemm_s8s8s8() driver passes the C8-only fused-store ABI to the
// original NN kernel symbol.  It does NOT execute AArch64 SME assembly.
#include "tests/reference.hpp"
#include "int8_gemm.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Production definitions live in int8_gemm.cpp, which this test links.
extern int nthreadsM;
extern int nthreadsN;

namespace {
constexpr int kM = 2048;
constexpr int kN = 2048;
constexpr int kK = fusion_test::kFixedK;
constexpr int kLda = kM + 7;
constexpr int kLdb = kK + 11;
constexpr int kLdc8 = kM + 23;
constexpr int8_t kC8Padding = static_cast<int8_t>(-91);
constexpr int8_t kKernelMarker = static_cast<int8_t>(91);

enum class C8Behavior { Inverse, Marker };

struct Invocation {
    int8_t* c8 = nullptr;
    unsigned mode = 0;
    C8Behavior c8_behavior = C8Behavior::Inverse;
    std::atomic<size_t> pack_a_calls{0};
    std::atomic<size_t> pack_b_calls{0};
    std::atomic<size_t> kernel_calls{0};
    std::atomic<bool> invalid{false};
};

Invocation* g_invocation = nullptr;

[[noreturn]] void fail(const std::string& message) {
    throw std::runtime_error(message);
}

void require(bool condition, const std::string& message) {
    if (!condition) fail(message);
}

int32_t expected_accumulator(int row, int col) {
    // Keep values far from padding and exercise positive/negative remainders.
    return static_cast<int32_t>(((row * 131 + col * 17 + 7) % 100003) - 50001);
}

bool tile_origin(const Int8FusedStoreParams* params, BLASLONG* row, BLASLONG* col) {
    if (g_invocation == nullptr || params == nullptr || params->c8 == nullptr) {
        return false;
    }
    const uintptr_t c8_begin = reinterpret_cast<uintptr_t>(g_invocation->c8);
    const uintptr_t c8_end = c8_begin + static_cast<uintptr_t>(kLdc8) * kN;
    const uintptr_t c8_address = reinterpret_cast<uintptr_t>(params->c8);
    if (c8_address < c8_begin || c8_address >= c8_end) return false;
    const size_t c8_offset = c8_address - c8_begin;
    *row = static_cast<BLASLONG>(c8_offset % kLdc8);
    *col = static_cast<BLASLONG>(c8_offset / kLdc8);
    return c8_offset == static_cast<size_t>(*row) +
                         static_cast<size_t>(*col) * kLdc8;
}

void check_full_team(int threads) {
    int observed = 0;
#pragma omp parallel num_threads(threads) reduction(max:observed)
    { observed = omp_get_num_threads(); }
    require(observed == threads,
            "portable production-driver test could not create the requested OpenMP team");
}

// Model the address calculation in INT8_INVERSE_SCALING_DIRECT exactly:
//
//   dst = c8_tile + ((pc + off * VL - c8_tile) >> 2)
//
// The original kernel moves pc through an INT32 (four-byte) virtual plane,
// while the direct store writes bytes.  Exercise non-four-byte C8 strides,
// all SAVE_ZACOL offsets, vector lengths representative of SME systems, tile
// origins and partial final vectors.  This is intentionally an integer-only
// test: the host cannot execute SME instructions, but it locks down the key
// coordinate conversion independently of the driver mock below.
void verify_virtual_word_to_c8_mapping() {
    constexpr uint64_t kSyntheticC8Base = UINT64_C(0x100000003);
    constexpr std::array<BLASLONG, 2> kMatrixSizes = {2048, 8192};
    constexpr std::array<BLASLONG, 5> kVectorBytes = {16, 32, 64, 128, 256};
    constexpr std::array<BLASLONG, 4> kRelativeRows = {0, 4, 64, 127};
    constexpr std::array<BLASLONG, 4> kRelativeCols = {0, 1, 31, 63};

    for (const BLASLONG matrix_m : kMatrixSizes) {
        const BLASLONG ldc8 = matrix_m + 23;  // Deliberately not a 4-byte stride.
        const std::array<BLASLONG, 5> tile_rows = {
            0, 64, 128, matrix_m / 2, matrix_m - 128};
        for (const BLASLONG matrix_n : kMatrixSizes) {
            const std::array<BLASLONG, 4> tile_cols = {
                0, 32, matrix_n / 2, matrix_n - 32};
            for (const BLASLONG tile_row : tile_rows) {
                for (const BLASLONG tile_col : tile_cols) {
                    const uint64_t c8_tile = kSyntheticC8Base +
                        static_cast<uint64_t>(tile_row) +
                        static_cast<uint64_t>(tile_col) * ldc8;
                    for (const BLASLONG vector_bytes : kVectorBytes) {
                        const BLASLONG int32_lanes = vector_bytes / 4;
                        for (const BLASLONG relative_row : kRelativeRows) {
                            for (const BLASLONG relative_col : kRelativeCols) {
                                if (tile_col + relative_col >= matrix_n) continue;

                                // pc is the same address that the legacy STNT1W
                                // would have used before its [pc, off, MUL VL]
                                // displacement was applied.
                                const uint64_t pc = c8_tile + 4 *
                                    (static_cast<uint64_t>(relative_row) +
                                     static_cast<uint64_t>(relative_col) * ldc8);
                                for (BLASLONG off = 0; off < 4; ++off) {
                                    const BLASLONG global_row = tile_row + relative_row +
                                        off * int32_lanes;
                                    if (global_row >= matrix_m) continue;
                                    const BLASLONG active_lanes =
                                        std::min(int32_lanes, matrix_m - global_row);

                                    const uint64_t helper_dst = c8_tile +
                                        ((pc + static_cast<uint64_t>(off) * vector_bytes -
                                          c8_tile) >> 2);
                                    const uint64_t expected_dst = kSyntheticC8Base +
                                        static_cast<uint64_t>(global_row) +
                                        static_cast<uint64_t>(tile_col + relative_col) * ldc8;
                                    require(helper_dst == expected_dst,
                                            "virtual word-to-C8 base-address mapping mismatch");

                                    // st1b uses a prefix byte predicate built from
                                    // the active S lanes.  Every active byte must
                                    // be contiguous at the expected C8 coordinate.
                                    for (BLASLONG lane = 0; lane < active_lanes; ++lane) {
                                        require(helper_dst + static_cast<uint64_t>(lane) ==
                                                    expected_dst + static_cast<uint64_t>(lane),
                                                "virtual word-to-C8 vector-lane mapping mismatch");
                                    }
                                    const uint64_t last_offset =
                                        helper_dst - kSyntheticC8Base + active_lanes - 1;
                                    require(last_offset / ldc8 ==
                                                static_cast<uint64_t>(tile_col + relative_col) &&
                                                    last_offset % ldc8 <
                                                static_cast<uint64_t>(matrix_m),
                                            "virtual word-to-C8 mapping touched C8 padding");
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void run_one(unsigned mode, int threads, C8Behavior c8_behavior) {
    check_full_team(threads);
    nthreadsM = threads;
    nthreadsN = 1;

    std::vector<int8_t> a(static_cast<size_t>(kLda) * kK, 3);
    std::vector<int8_t> b(static_cast<size_t>(kLdb) * kN, -5);
    const auto original_a = a;
    const auto original_b = b;
    std::vector<int8_t> sa(static_cast<size_t>(kK) * 128 * 32 + 65536, 0);
    std::vector<int8_t> sb(static_cast<size_t>(kK) * 8192 + 65536, 0);
    std::vector<int8_t> c8(static_cast<size_t>(kLdc8) * kN, kC8Padding);
    Invocation invocation;
    invocation.c8 = c8.data();
    invocation.mode = mode;
    invocation.c8_behavior = c8_behavior;
    g_invocation = &invocation;
    cblas_gemm_s8s8s8(
        CblasColMajor, CblasNoTrans, CblasNoTrans,
        kM, kN, kK, 1.0f, a.data(), kLda, 0, b.data(), kLdb, 0, 0.0f,
        sa.data(), sb.data(), c8.data(), kLdc8, mode);
    g_invocation = nullptr;

    require(!invocation.invalid.load(),
            "production driver passed an invalid fused-store parameter block");
    const size_t expected_m_tiles = threads == 1 ? 16 : 32;
    const size_t expected_kernel_calls = expected_m_tiles * (kN / 32);
    require(invocation.pack_a_calls.load() == expected_m_tiles,
            "unexpected A-pack call count");
    require(invocation.pack_b_calls.load() == static_cast<size_t>(threads),
            "unexpected B-pack call count");
    require(invocation.kernel_calls.load() == expected_kernel_calls,
            "unexpected production-kernel call count");

    for (int col = 0; col < kN; ++col) {
        for (int row = 0; row < kM; ++row) {
            const int32_t accumulator = expected_accumulator(row, col);
            const int8_t expected8 = c8_behavior == C8Behavior::Inverse
                ? fusion_test::inverse_integer(accumulator, mode)
                : kKernelMarker;
            const size_t c8_index = static_cast<size_t>(col) * kLdc8 + row;
            if (c8[c8_index] != expected8) {
                fail("mocked C8-only production output mismatch mode=" +
                     std::to_string(mode) + " threads=" + std::to_string(threads) +
                     " row=" + std::to_string(row) + " col=" + std::to_string(col));
            }
        }
        for (int row = kM; row < kLdc8; ++row) {
            if (c8[static_cast<size_t>(col) * kLdc8 + row] != kC8Padding) {
                fail("production driver mock touched C8 padding");
            }
        }
    }
    require(a == original_a && b == original_b,
            "production driver modified an input matrix");
}
}  // namespace

extern "C" void int8_sme_gemm_itcopy(
    const BLASLONG, const BLASLONG, const BLASINT8*, const BLASLONG,
    BLASINT8*, BLASINT8) {
    if (g_invocation != nullptr) ++g_invocation->pack_a_calls;
}

extern "C" void int8_sme_gemm_oncopy(
    const BLASLONG, const BLASLONG, const BLASINT8*, const BLASLONG,
    BLASINT8*, BLASINT8) {
    if (g_invocation != nullptr) ++g_invocation->pack_b_calls;
}

extern "C" void int8_sme_gemm_kernel_nn(
    BLASLONG rows, BLASLONG cols, BLASLONG k, void*, BLASLONG, float alpha,
    void*, BLASLONG, int8_t* c8, BLASLONG ldc8, void* buf) {
    Invocation* invocation = g_invocation;
    const auto* params = static_cast<const Int8FusedStoreParams*>(buf);
    BLASLONG base_row = 0;
    BLASLONG base_col = 0;
    const int32_t expected_modulus = invocation != nullptr && invocation->mode != 0
        ? fusion_test::kModuli[invocation->mode - 1]
        : 0;
    const uint32_t expected_magic = invocation != nullptr && invocation->mode != 0
        ? fusion_test::kReciprocalMagic[invocation->mode - 1]
        : 0u;
    if (invocation == nullptr || params == nullptr || k != kK || alpha != 1.0f ||
        params->c8 != c8 || ldc8 != kLdc8 ||
        params->modulus != expected_modulus ||
        params->reciprocal_magic != expected_magic || rows <= 0 || cols <= 0 ||
        !tile_origin(params, &base_row, &base_col) || base_row + rows > kM ||
        base_col + cols > kN) {
        if (invocation != nullptr) invocation->invalid.store(true);
        return;
    }

    for (BLASLONG col = 0; col < cols; ++col) {
        for (BLASLONG row = 0; row < rows; ++row) {
            const int32_t value = expected_accumulator(static_cast<int>(base_row + row),
                                               static_cast<int>(base_col + col));
            params->c8[col * ldc8 + row] =
                invocation->c8_behavior == C8Behavior::Inverse
                    ? fusion_test::inverse_integer(value, invocation->mode)
                    : kKernelMarker;
        }
    }
    ++invocation->kernel_calls;
}

int main() {
    try {
        verify_virtual_word_to_c8_mapping();
        // All modes validate the C++ mode-to-modulus wiring on the serial path.
        for (unsigned mode = 0; mode <= fusion_test::kModuli.size(); ++mode) {
            run_one(mode, 1, C8Behavior::Inverse);
        }
        // Exercise production tiling and barriers with the target thread layout.
        for (unsigned mode : {0u, 1u, 19u}) {
            run_one(mode, 32, C8Behavior::Inverse);
        }
        // The mock imitates the assembly C8 store with a non-oracle marker.  If
        // a C++ inverse-scaling loop still runs after the kernel returns, this
        // marker would be overwritten and this call fails.
        run_one(1, 1, C8Behavior::Marker);
        std::cout << "PASS host production-driver mock: virtual word-to-C8 address "
                     "mapping, C8-only API wiring, all modulus/magic pairs, C8 stride, padding, "
                     "1/32-thread tiling, and absence of a post-kernel C++ "
                     "inverse-scaling pass verified.\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL host production-driver mock: " << error.what() << '\n';
        return 1;
    }
}
