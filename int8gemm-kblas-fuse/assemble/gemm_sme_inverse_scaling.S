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
 * This is the restored integer SDIV implementation.  It intentionally reads
 * only c8 and modulus; the currently retained inv_p/neg_p ABI fields are not
 * consumed by this epilogue.
 */

#define INVERSE_SCALING_PARAMS_C8       0
#define INVERSE_SCALING_PARAMS_MODULUS  8
#define INVERSE_SCALING_PARAMS_INV_P    16
#define INVERSE_SCALING_PARAMS_NEG_P    24

/*
 * Stage A baseline: KERNEL_SAVE_C_* already invokes PREPARE at each completed
 * ZA save tile.  Keep that call site intact while restoring the legacy direct
 * SDIV path below, which still loads its scalar state per output vector.
 */
.macro INT8_INVERSE_SCALING_PREPARE
.endm

/*
 * Apply inverse scaling to one INT32 vector which has just been read from ZA,
 * then store its INT8 result directly to C8.
 *
 * For modulus == 0, the operation is the low eight bits of each INT32 lane.
 * For every supported positive odd modulus p, the nonzero path computes the
 * centered signed residue using exact integer arithmetic:
 *
 *     q = trunc(value / p)
 *     r = value - q * p
 *     if (r >  floor(p/2)) r -= p
 *     if (r < -floor(p/2)) r += p
 *
 * Odd p means no integer input can be exactly halfway between two multiples,
 * so this is equivalent to the reference round-to-nearest inverse scaling.
 *
 * The fused NN build keeps the established kernel C/ldc traversal as a
 * virtual INT32 coordinate plane.  Its numeric base is the C8 tile origin,
 * but it is never dereferenced as C32.  Dividing its byte displacement by
 * sizeof(int32_t) maps it to the real byte displacement within C8:
 *
 *   dst = c8_tile + ((pc + off * VL) - c8_tile) / sizeof(int32_t)
 *
 * x4/x5/x6 are scratch after a completed tile reaches SAVE_ZACOL.  The
 * kernel's virtual C base and stride remain in their ordinary registers and
 * are not reused here.  p5/p6 are temporary save-stage predicates; p0-p4/p7
 * traversal state and p1/p2 MOVA predicates are left intact.  SAVE_ZACOL
 * passes tmp_q and tmp_mod disjoint from value in every production schedule.
 */
.macro INT8_INVERSE_SCALING_DIRECT pc,off,reg_pg,value,tmp_q,tmp_mod
    // The existing 11th kernel argument is the fused-store parameter block.
    // SAVE_REGS subtracts 11 * 16 bytes, so the incoming stack argument is
    // at sp + (11 * 16 + 8).
    ldr     x4, [sp, #(11 * 16 + 8)]
    cbz     x4, 9f

    // x4 becomes the destination address below.  c8 also provides the null
    // store sentinel for an invalid/disabled fused destination.
    ldr     x5, [x4, #INVERSE_SCALING_PARAMS_C8]
    cbz     x5, 9f
    ldr     w6, [x4, #INVERSE_SCALING_PARAMS_MODULUS]

    // Build the final byte predicate while x4 is still free.  p6 is not a
    // source predicate for this save invocation (reg_pg is p1 or p2).
    cntp    x4, \reg_pg, \reg_pg\().s
    whilelt p6.b, xzr, x4

    // Map the virtual INT32 save address to the byte-addressed C8 tile.
    addvl   x4, \pc, \off
    sub     x4, x4, x5
    lsr     x4, x4, #2
    add     x4, x5, x4

    // Mode zero is the established low-byte path and must not execute SDIV.
    cbz     w6, 8f

    // q = trunc(value / p).  SDIV uses a destructive numerator, so preserve
    // the source value in tmp_q.  tmp_mod contains a broadcast p.
    dup     \tmp_mod\().s, w6
    mov     \tmp_q\().d, \value\().d
    sdiv    \tmp_q\().s, \reg_pg\()/M, \tmp_q\().s, \tmp_mod\().s

    // value <- value - q * p.
    mls     \value\().s, \reg_pg\()/M, \tmp_q\().s, \tmp_mod\().s

    // Shift the truncation remainder into the centered interval.
    lsr     w5, w6, #1                  // floor(p / 2), p is odd
    dup     \tmp_q\().s, w5
    cmpgt   p5.s, \reg_pg\()/z, \value\().s, \tmp_q\().s
    sub     \value\().s, p5/m, \value\().s, \tmp_mod\().s

    neg     \tmp_q\().s, \reg_pg\()/M, \tmp_q\().s
    cmpgt   p5.s, \reg_pg\()/z, \tmp_q\().s, \value\().s
    add     \value\().s, p5/m, \value\().s, \tmp_mod\().s

8:
    // Keep the low byte of each corrected INT32 lane and write C8 directly.
    uzp1    \value\().h, \value\().h, \value\().h
    uzp1    \value\().b, \value\().b, \value\().b
    st1b    { \value\().b }, p6, [x4]
9:
.endm
