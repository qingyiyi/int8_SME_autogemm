/*
 * Production INT8 inverse-scaling epilogue.
 *
 * The NN GEMM kernel leaves the INT32 accumulators in ZA until each result
 * vector is extracted.  The direct helper below inverse-scales that MOVA
 * vector and writes INT8 immediately, without materializing a C32 tile.
 *
 * Int8FusedStoreParams layout (bytes):
 *   0  c8       8  modulus       12 reciprocal_magic
 */

#define INVERSE_SCALING_PARAMS_C8                0
#define INVERSE_SCALING_PARAMS_MODULUS           8
#define INVERSE_SCALING_PARAMS_RECIPROCAL_MAGIC 12

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
 * The quotient uses an exact fixed-reciprocal sequence.  The ABI carries
 * m32=floor(2^32/p); this helper derives m31=m32>>1=floor(2^31/p), then
 * computes:
 *
 *     q0 = SQDMULH(value, m31) = floor(value*m31 / 2^31)
 *     r  = value - q0*p
 *     if (r < 0)  r += p
 *     if (r >= p) r -= p
 *
 * For every supported p and signed INT32 input, q0 differs from floor(value/p)
 * by at most one.  The two one-step normalizations therefore recover the exact
 * [0,p) remainder before the final centered-remainder correction.  SQDMULH is
 * non-saturating for this operand range (and for the fixed K=2048 accumulator
 * bound), so no FP32/FP64 conversion or division is involved.
 *
 * In the fused NN build, the existing kernel C/ldc argument slots are kept as
 * a virtual INT32 coordinate plane so the proven GEMM traversal is unchanged.
 * The numeric base is the same as `c8`; the virtual plane is never dereferenced.
 * Converting the virtual byte displacement to an INT32 element displacement
 * therefore gives the real C8 byte address directly, without loading `ldc8`.
 *
 * x4/x5/x6 are dead after GEMM reaches SAVE_ZACOL.  At the beginning of a
 * completed ZA result tile, PREPARE caches the C8 base in x5 (or zero when
 * `buf` is null), the scalar modulus in w6, and the fixed reciprocal vectors
 * in z27/z28/z29.  x4 is then free as a per-vector count/address scratch.  This
 * leaves no parameter-stack, C8-base, or modulus load in the hot direct helper.
 * p5/p6 are temporary predicates: MOPA predicates for the just-finished M
 * block, rebuilt before a later M block can accumulate.  The live p0-p4/p7
 * traversal predicates and the p1/p2 MOVA predicates are left intact.
 * SAVE_ZACOL passes one quotient scratch Z register (`tmp_q`) which is disjoint
 * from the current value register in every existing INT8 save-register schedule.
 *
 * The fixed reciprocal constants are invariant for one kernel invocation.  The
 * save entry below constructs them once per completed ZA result tile in
 * z27/z28/z29.  Those registers are dead throughout every current save schedule
 * and are overwritten again before the following accumulation tile.  Keeping
 * this setup outside the per-vector helper removes repeated parameter loads,
 * shifts and DUPs without changing the value/address/predicate path below.
 */
.macro INT8_INVERSE_SCALING_PREPARE
    // The 11th argument (buf) is at incoming sp+8; SAVE_REGS subtracts 176.
    // x5 is also the per-vector store-enable sentinel, so null buf cannot
    // inherit the old origPB value that occupied x5 before SAVE_ZACOL.
    mov     x5, xzr
    ldr     x4, [sp, #(11 * 16 + 8)]
    cbz     x4, 9f

    // Cache scalar state once for every completed ZA result tile.  x4 is free
    // after these loads and is reused by the direct helper as its scratch.
    ldr     x5, [x4, #INVERSE_SCALING_PARAMS_C8]
    ldr     w6, [x4, #INVERSE_SCALING_PARAMS_MODULUS]

    // Mode 0 takes the direct helper's original low-byte fast path, so do not
    // construct reciprocal vectors for it.
    cbz     w6, 9f

    ldr     w4, [x4, #INVERSE_SCALING_PARAMS_RECIPROCAL_MAGIC]
    lsr     w4, w4, #1                  // floor(2^31 / p)
    dup     z27.s, w4                   // magic31
    dup     z28.s, w6                   // modulus
    lsr     w4, w6, #1                  // floor(p/2), p is odd
    dup     z29.s, w4                   // half modulus
9:
.endm

.macro INT8_INVERSE_SCALING_DIRECT pc,off,reg_pg,value,tmp_q
    // PREPARE cleared x5 before inspecting buf, so it is a stable null-buf
    // guard as well as the cached C8 base for the whole save block.
    cbz     x5, 9f

    // Narrowing an active prefix of S lanes produces one contiguous B lane
    // per original INT32 lane.  Build the matching byte predicate before x4
    // is reused as the destination-address scratch.
    cntp    x4, \reg_pg, \reg_pg\().s
    whilelt p6.b, xzr, x4

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

    // A zero modulus is the low-byte mode and must not enter the reciprocal
    // path, so mode 0 retains its original fast epilogue cost.
    cbz     w6, 8f

    // z27/z28/z29 were prepared once for this completed ZA tile.  The
    // mode-0 branch remains before all inverse arithmetic, preserving its
    // original low-byte fast path.
    sqdmulh \tmp_q\().s, \value\().s, z27.s

    // r = value - q0*p.  q0 can be one quotient below/above floor(value/p),
    // depending on the sign; normalize it to the unique unsigned remainder.
    mls     \value\().s, \reg_pg\()/M, \tmp_q\().s, z28.s
    cmplt   p5.s, \reg_pg\()/z, \value\().s, #0
    add     \value\().s, p5/m, \value\().s, z28.s
    cmpge   p5.s, \reg_pg\()/z, \value\().s, z28.s
    sub     \value\().s, p5/m, \value\().s, z28.s

    // r is now the exact [0,p) remainder, so only the positive centered-
    // remainder correction is required.
    cmpgt   p5.s, \reg_pg\()/z, \value\().s, z29.s
    sub     \value\().s, p5/m, \value\().s, z28.s

8:
    // Keep the low byte of each (possibly corrected) INT32 lane.
    uzp1    \value\().h, \value\().h, \value\().h
    uzp1    \value\().b, \value\().b, \value\().b
    st1b    { \value\().b }, p6, [x4]
9:
.endm
