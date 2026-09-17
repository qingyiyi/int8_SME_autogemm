#ifndef INT8_FUSION_TEST_REFERENCE_HPP
#define INT8_FUSION_TEST_REFERENCE_HPP

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace fusion_test {

constexpr int kFixedK = 2048;
// Frozen independently of production MODULI_I: do not import that table here.
constexpr std::array<int32_t, 19> kModuli = {
    255, 253, 251, 247, 241, 239, 233, 229, 227, 223,
    217, 211, 199, 197, 193, 191, 181, 179, 173
};
// Frozen independently of the production dispatch table.  Each entry is
// floor(2^32 / p) for the modulus at the same index.
constexpr std::array<uint32_t, 19> kReciprocalMagic = {
    16843009u, 16976155u, 17111423u, 17388531u, 17821441u,
    17970574u, 18433336u, 18755315u, 18920560u, 19259943u,
    19792476u, 20355295u, 21582750u, 21801864u, 22253716u,
    22486739u, 23729101u, 23994230u, 24826400u
};
constexpr int32_t kMaximumAccumulatorMagnitude = kFixedK * 128 * 128;

constexpr bool reciprocal_magic_is_exact() {
    for (size_t i = 0; i < kModuli.size(); ++i) {
        if (kReciprocalMagic[i] !=
            (UINT64_C(1) << 32) / static_cast<uint32_t>(kModuli[i])) {
            return false;
        }
    }
    return true;
}
static_assert(reciprocal_magic_is_exact(),
              "frozen reciprocal table must equal floor(2^32 / p)");

inline int8_t low_byte(int32_t value) {
    const int byte = static_cast<uint32_t>(value) & 255u;
    // Avoid implementation-defined out-of-range signed narrowing in the oracle.
    return static_cast<int8_t>(byte < 128 ? byte : byte - 256);
}

inline void check_mode(unsigned mode) {
    if (mode > kModuli.size()) {
        throw std::invalid_argument("num_moduli must be in [0, 19]");
    }
}

// Frozen copy of the current scalar semantics. Caller must use FE_TONEAREST.
inline int8_t inverse_fp64(int32_t value, unsigned mode) {
    check_mode(mode);
    if (mode == 0) return low_byte(value);
    const double p = static_cast<double>(kModuli[mode - 1]);
    const double reciprocal = 1.0 / p;
    const double v = static_cast<double>(value);
    const double q = std::rint(v * reciprocal);
    const double remainder = std::fma(-p, q, v);
    if (remainder < -128 || remainder > 127 || std::trunc(remainder) != remainder) {
        throw std::runtime_error("FP64 reference produced a non-int8 remainder");
    }
    return static_cast<int8_t>(remainder);
}

// Independent integer oracle, not a proposed production optimization.
// Every listed modulus is odd, so an integer input cannot be a half-modulus tie.
inline int8_t inverse_integer(int32_t value, unsigned mode) {
    check_mode(mode);
    if (mode == 0) return low_byte(value);
    const int64_t p = kModuli[mode - 1];
    int64_t remainder = static_cast<int64_t>(value) % p;
    if (remainder > p / 2) remainder -= p;
    if (remainder < -p / 2) remainder += p;
    return static_cast<int8_t>(remainder);
}

// Host model of the production scalar inverse-scaling epilogue.  It deliberately
// follows `sdiv` + `msub` and the two signed correction branches rather than
// using `%`, so this test catches a future change in the assembly algorithm.
inline int8_t inverse_fused_scalar_model(int32_t value, unsigned mode) {
    check_mode(mode);
    if (mode == 0) return low_byte(value);
    const int32_t p = kModuli[mode - 1];
    const int32_t half = p >> 1;
    const int32_t quotient = value / p;  // AArch64 SDIV truncates toward zero.
    int32_t remainder = value - quotient * p;  // AArch64 MSUB result.
    if (remainder > half) {
        remainder -= p;
    } else if (remainder + half < 0) {
        remainder += p;
    }
    return static_cast<int8_t>(remainder);
}

// Host model of the proposed division-free epilogue.  For B=2^32 and
// magic=floor(B/p), high32(abs(value)*magic) never overestimates abs(value)/p.
// For every signed INT32 magnitude its error is less than one quotient, so one
// `remainder >= p` correction recovers the exact non-negative remainder.
inline int8_t inverse_magic_scalar_model(int32_t value, unsigned mode) {
    check_mode(mode);
    if (mode == 0) return low_byte(value);

    const uint32_t p = static_cast<uint32_t>(kModuli[mode - 1]);
    const uint64_t magnitude = value < 0
        ? static_cast<uint64_t>(-static_cast<int64_t>(value))
        : static_cast<uint64_t>(value);
    const uint64_t quotient =
        (magnitude * static_cast<uint64_t>(kReciprocalMagic[mode - 1])) >> 32;
    uint64_t remainder = magnitude - quotient * p;
    if (remainder >= p) remainder -= p;

    int32_t centered = static_cast<int32_t>(remainder);
    if (centered > static_cast<int32_t>(p >> 1)) {
        centered -= static_cast<int32_t>(p);
    }
    if (value < 0) centered = -centered;
    return static_cast<int8_t>(centered);
}

// One independently computed element of a column-major NN GEMM.  The target
// acceptance test uses this bounded oracle at tile corners rather than trying
// to materialize a scalar M*N*K reference for the 2048/8192 business shapes.
inline int32_t accumulator_reference_element(
    const int8_t* a, int lda, const int8_t* b, int ldb, int row, int col) {
    int64_t sum = 0;
    for (int l = 0; l < kFixedK; ++l) {
        sum += static_cast<int64_t>(a[static_cast<size_t>(l) * lda + row]) *
               b[static_cast<size_t>(col) * ldb + l];
    }
    if (sum < std::numeric_limits<int32_t>::min() ||
        sum > std::numeric_limits<int32_t>::max()) {
        throw std::overflow_error("reference GEMM exceeds int32 range");
    }
    return static_cast<int32_t>(sum);
}


}  // namespace fusion_test
#endif
