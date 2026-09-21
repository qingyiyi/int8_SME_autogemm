/*
 * Production INT8 inverse-scaling epilogue.
 *
 * The NN GEMM kernel leaves the INT32 accumulators in ZA until each result
 * vector is extracted.  The direct helper below inverse-scales that MOVA
 * vector and writes INT8 immediately, without materializing a C32 tile.
 *
 * Int8FusedStoreParams layout (bytes):
 *   0  c8       8  modulus       12 reserved
 *  16  inv_p   24  neg_p         32-byte object, 16-byte aligned
 *
 * The restored integer SDIV epilogue consumes only c8 and modulus.  inv_p and
 * neg_p are retained in the C++ ABI during this staged optimization, but are
 * deliberately not read by the assembly path.
 */

#define INVERSE_SCALING_PARAMS_C8       0
#define INVERSE_SCALING_PARAMS_MODULUS  8
#define INVERSE_SCALING_PARAMS_INV_P    16
#define INVERSE_SCALING_PARAMS_NEG_P    24

/*
 * Load invariant direct-store state once for each completed ZA save tile.
 *
 * The 11th kernel argument (buf) is at incoming sp+8; SAVE_REGS subtracts
 * 11 * 16 bytes.  The GEMM compute phase owns its normal working registers,
 * so this runs only after the tile's MOPA accumulation is complete.
 *
 * Cached state:
 *   x5      C8 tile base and null-store sentinel
 *   w6      modulus/mode: 0 = low byte; nonzero = centered-remainder SDIV
 *   z27.s   broadcast modulus p
 *   z28.s   broadcast floor(p / 2)
 *
 * x5/w6 are reset first, so a null parameter block, null C8 base, or zero
 * modulus cannot retain state from the preceding tile.  z27/z28 are unused by
 * the production save-register schedule, and p0-p4/p7 plus p1/p2 are left
 * untouched.  The target assembler accepts low predicate registers in SVE
 * memory operations, so the mode is held in a GPR rather than a high
 * predicate register.
 */
.macro INT8_INVERSE_SCALING_PREPARE
    mov     x5, xzr
    mov     w6, wzr

    ldr     x4, [sp, #(11 * 16 + 8)]
    cbz     x4, 9f

    ldr     x5, [x4, #INVERSE_SCALING_PARAMS_C8]
    cbz     x5, 9f

    ldr     w6, [x4, #INVERSE_SCALING_PARAMS_MODULUS]
    cbz     w6, 9f

    // The scalar modulus and half-modulus are invariant for every vector in
    // this ZA tile.  Broadcast them once rather than loading/duplicating them
    // from INT8_INVERSE_SCALING_DIRECT for every MOVA result.
    dup     z27.s, w6
    lsr     w4, w6, #1
    dup     z28.s, w4

    // Keep w6 as the nonzero-mode test for every vector in this tile.  x4 is
    // scratch again after this preparation macro returns.
9:
.endm

/*
 * Apply inverse scaling to one INT32 vector which has just been read from ZA,
 * then store its INT8 result directly to C8.
 *
 * For modulus == 0, the operation is the low eight bits of each INT32 lane.
 * For every supported positive odd modulus p, the nonzero path computes the
 * exact centered signed residue:
 *
 *     q = trunc(value / p)
 *     r = value - q * p
 *     if (r >  floor(p/2)) r -= p
 *     if (r < -floor(p/2)) r += p
 *
 * Because all supported p are odd, no integer input is exactly halfway between
 * two multiples of p.  This integer form is therefore equivalent to the
 * reference round-to-nearest inverse scaling, without FP64 conversion or
 * rounding instructions.
 *
 * The fused NN build preserves the established kernel C/ldc traversal as a
 * virtual INT32 coordinate plane.  Its numeric base equals the C8 tile base,
 * but it is never dereferenced as C32.  The virtual byte displacement divided
 * by sizeof(int32_t) is the real byte displacement in C8:
 *
 *   dst = c8_tile + ((pc + off * VL) - c8_tile) / sizeof(int32_t)
 *
 * PREPARE keeps x5/w6/z27/z28 stable across all direct stores in the tile.
 * x4 is per-vector scratch after a completed tile reaches SAVE_ZACOL.
 * p5 is the centered-remainder comparison predicate and p6 is the byte-store
 * predicate.  p0-p4/p7 traversal state and p1/p2 MOVA predicates remain
 * intact.  tmp_q is disjoint from value in every production save schedule;
 * tmp_mod is retained in the macro signature for existing call sites but is
 * no longer needed because z27.s already contains p.
 */
.macro INT8_INVERSE_SCALING_DIRECT pc,off,reg_pg,value,tmp_q,tmp_mod
    // x5 is cleared by PREPARE for a null/disabled fused destination.
    cbz     x5, 9f

    // Build the eventual contiguous C8 store predicate before x4 is reused as
    // the virtual-address / destination-address scratch.  reg_pg is p1 or p2,
    // never p6, in all production SAVE_ZACOL invocations.
    cntp    x4, \reg_pg, \reg_pg\().s
    whilelt p6.b, xzr, x4

    // Map the virtual INT32 save address to the byte-addressed C8 tile.
    addvl   x4, \pc, \off
    sub     x4, x4, x5
    lsr     x4, x4, #2
    add     x4, x5, x4

    // Modulus zero is the established low-byte fast path and must not execute
    // SDIV or the centered-remainder arithmetic.
    cbz     w6, 8f

    // q = trunc(value / p).  SDIV's numerator is destructive, so tmp_q first
    // receives the MOVA value.  z27.s was broadcast once by PREPARE.
    mov     \tmp_q\().d, \value\().d
    sdiv    \tmp_q\().s, \reg_pg\()/M, \tmp_q\().s, z27.s

    // value <- value - q * p.
    mls     \value\().s, \reg_pg\()/M, \tmp_q\().s, z27.s

    // Correct the truncation remainder into [-floor(p/2), floor(p/2)].
    cmpgt   p5.s, \reg_pg\()/z, \value\().s, z28.s
    sub     \value\().s, p5/m, \value\().s, z27.s

    // tmp_q <- -floor(p/2), then correct overly negative remainders.  Keep
    // the MOV/NEG sequence because the target assembler requires the
    // predicated NEG source and destination to match.
    mov     \tmp_q\().d, z28.d
    neg     \tmp_q\().s, \reg_pg\()/M, \tmp_q\().s
    cmpgt   p5.s, \reg_pg\()/z, \tmp_q\().s, \value\().s
    add     \value\().s, p5/m, \value\().s, z27.s

8:
    // The supported centered residue always fits INT8.  Compact one byte per
    // original INT32 lane and write it directly; no C32 tile is materialized.
    uzp1    \value\().h, \value\().h, \value\().h
    uzp1    \value\().b, \value\().b, \value\().b
    st1b    { \value\().b }, p6, [x4]
9:
.endm
