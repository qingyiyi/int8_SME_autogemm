/*
 * Production INT8 inverse-scaling epilogue.
 *
 * The NN GEMM kernel leaves the INT32 accumulators in ZA until each result
 * vector is extracted.  The direct helper below inverse-scales that MOVA
 * vector and writes INT8 immediately, without materializing a C32 tile.
 *
 * Int8FusedStoreParams layout (bytes):
 *   0  c8       8  modulus       12 reserved
 */

#define INVERSE_SCALING_PARAMS_C8       0
#define INVERSE_SCALING_PARAMS_MODULUS  8

/*
 * Apply inverse scaling to one INT32 vector which has just been read from ZA,
 * then store its INT8 result directly to C8.
 *
 * For modulus == 0, the operation is the low eight bits of each INT32 lane.
 * For a supported nonzero modulus p (all p are positive odd values), the
 * centered remainder is computed as:
 *
 *     q = trunc(value / p)
 *     r = value - q * p
 *     if (r >  p/2) r -= p
 *     if (r < -p/2) r += p
 *
 * This is the same integer form used by the validated scalar epilogue and is
 * equivalent to the old FP64 rint/fma result for the supported odd moduli.
 *
 * In the fused NN build, the existing kernel C/ldc argument slots are kept as
 * a virtual INT32 coordinate plane so the proven GEMM traversal is unchanged.
 * The numeric base is the same as `c8`; the virtual plane is never dereferenced.
 * Converting the virtual byte displacement to an INT32 element displacement
 * therefore gives the real C8 byte address directly, without loading `ldc8`.
 *
 * x4/x5/x6 are dead after GEMM reaches SAVE_ZACOL.  p5/p6 are temporary
 * predicates: MOPA predicates for the just-finished M block, rebuilt before a
 * later M block can accumulate.  The live p0-p4/p7 traversal predicates and the
 * p1/p2 MOVA predicates are left intact.  SAVE_ZACOL passes two scratch Z
 * registers (`tmp_q` and `tmp_mod`) which are disjoint from the current value
 * register in every existing save-register schedule.
 */
.macro INT8_INVERSE_SCALING_DIRECT pc,off,reg_pg,value,tmp_q,tmp_mod
    // The 11th argument (buf) is at incoming sp+8; SAVE_REGS subtracts 176.
    ldr     x4, [sp, #(11 * 16 + 8)]
    cbz     x4, 9f

    // x4 becomes the destination address below.  x4/x5/x6 originally hold
    // LDA/origPB/LDB and are dead once GEMM reaches SAVE_ZACOL.
    ldr     w6, [x4, #INVERSE_SCALING_PARAMS_MODULUS]
    ldr     x5, [x4, #INVERSE_SCALING_PARAMS_C8]

    // The proven GEMM traversal presents a virtual INT32 address plane whose
    // numeric base is c8_tile.  Converting its byte displacement to an
    // element displacement maps it directly onto the byte-addressed C8 tile;
    // no row/column divide and no ldc8 load are required:
    //
    //   dst = c8_tile + ((pc + off * VL) - c8_tile) / sizeof(int32_t)
    addvl   x4, \pc, \off
    sub     x4, x4, x5
    lsr     x4, x4, #2
    add     x4, x5, x4

    // Narrowing an active prefix of S lanes produces one contiguous B lane
    // per original INT32 lane.  Build the matching byte predicate once.
    cntp    x5, \reg_pg, \reg_pg\().s
    whilelt p6.b, xzr, x5

    // A zero modulus is the low-byte mode and must not execute SDIV.
    cbz     w6, 8f

    // q = trunc(value / p), with SDIV's required destructive numerator.
    dup     \tmp_mod\().s, w6
    mov     \tmp_q\().d, \value\().d
    sdiv    \tmp_q\().s, \reg_pg\()/M, \tmp_q\().s, \tmp_mod\().s

    // value <- value - q*p (MLS is the predicated destructive form).
    mls     \value\().s, \reg_pg\()/M, \tmp_q\().s, \tmp_mod\().s

    // Correct the signed remainder into the centered interval.
    // x5 is no longer needed after WHILELT, so reuse it for floor(p/2).
    lsr     w5, w6, #1                  // floor(p/2), p is odd
    dup     \tmp_q\().s, w5
    cmpgt   p5.s, \reg_pg\()/z, \value\().s, \tmp_q\().s
    sub     \value\().s, p5/m, \value\().s, \tmp_mod\().s

    neg     \tmp_q\().s, \reg_pg\()/M, \tmp_q\().s
    cmpgt   p5.s, \reg_pg\()/z, \tmp_q\().s, \value\().s
    add     \value\().s, p5/m, \value\().s, \tmp_mod\().s

8:
    // Keep the low byte of each (possibly corrected) INT32 lane.
    uzp1    \value\().h, \value\().h, \value\().h
    uzp1    \value\().b, \value\().b, \value\().b
    st1b    { \value\().b }, p6, [x4]
9:
.endm
