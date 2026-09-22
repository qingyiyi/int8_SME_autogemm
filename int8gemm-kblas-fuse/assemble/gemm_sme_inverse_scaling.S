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
 * P1 fixed-full-tile fast path: KERNEL_SAVE_C_* invokes PREPARE once for each
 * completed ZA save tile.  Under the production contract (M/N are multiples
 * of 2048 and SVL is 64 bytes), every saved .s vector has exactly 16 active
 * INT32 lanes.  After the two UZP1 compactions, its 16 C8 results occupy only
 * bytes [0, 15] of the Z register.  Build that fixed store predicate once per
 * tile instead of repeating CNT P + WHILELT for every MOVA result.
 *
 * Do not replace this with PTRUE p6.b: that would activate all 64 bytes and
 * would store packed garbage beyond the 16 valid C8 results.  Partial-vector
 * handling remains intentionally outside the fixed production contract.
 *
 * x4 is save-stage scratch at this point.  p6 is not used by the following
 * SAVE_ZACOL calls as an input predicate (they use p1/p2), and is rebuilt by
 * the GEMM traversal before the next MOPA phase.
 */
.macro INT8_INVERSE_SCALING_PREPARE
    mov     x4, #16
    whilelt p6.b, xzr, x4
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
 * P3 keeps the fused kernel's C cursor and LDC in direct C8 byte coordinates:
 * `pc` already identifies the first packed C8 result for this ZA vector group.
 * The fixed production SME vector length is 64 bytes, so a .s vector has 16
 * lanes and, after UZP1 x 2, its compacted C8 result occupies 16 bytes.  Thus
 * SAVE_ZACOL offsets 0..3 map directly to pc + {0, 16, 32, 48}; they must not
 * be used as MUL VL byte offsets, which would incorrectly step by 64 bytes.
 *
 * x4/x5/x6 are scratch after a completed tile reaches SAVE_ZACOL.  p5/p6 are
 * temporary save-stage predicates; p0-p4/p7 traversal state and p1/p2 MOVA
 * predicates are left intact.  SAVE_ZACOL passes tmp_q and tmp_mod disjoint
 * from value in every production schedule.
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

    // P1 PREPARE already made p6.b active for exactly the low 16 packed C8
    // bytes.  reg_pg is p1 or p2 in production; all 16 .s lanes are active
    // by the fixed M/N=2048-multiple contract.  `pc` itself is the direct C8
    // cursor, so there is deliberately no C32-to-C8 address conversion here.

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

    // P3 direct C8 cursor: the fixed 64-byte SVL produces 16 packed bytes per
    // ZA .s vector.  Keep offsets explicit so off cannot accidentally become
    // an off * VL (64-byte) address calculation.
.if \off == 0
    st1b    { \value\().b }, p6, [\pc]
.elseif \off == 1
    add     x4, \pc, #16
    st1b    { \value\().b }, p6, [x4]
.elseif \off == 2
    add     x4, \pc, #32
    st1b    { \value\().b }, p6, [x4]
.elseif \off == 3
    add     x4, \pc, #48
    st1b    { \value\().b }, p6, [x4]
.else
    .error "unsupported fused C8 SAVE_ZACOL offset"
.endif
9:
.endm
