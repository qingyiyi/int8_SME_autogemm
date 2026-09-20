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
 * B1 populates inv_p/neg_p in the C++ driver.  B2 consumes those FP64 fields
 * here, so the GEMM call never performs the reciprocal division.
 */

#define INVERSE_SCALING_PARAMS_C8       0
#define INVERSE_SCALING_PARAMS_MODULUS  8
#define INVERSE_SCALING_PARAMS_INV_P    16
#define INVERSE_SCALING_PARAMS_NEG_P    24

/*
 * Apply inverse scaling to one INT32 vector which has just been read from ZA,
 * then store its INT8 result directly to C8.
 *
 * For modulus == 0, the operation is the low eight bits of each INT32 lane.
 * For a supported nonzero modulus p (all p are positive odd values), this is
 * the original FP64 operation:
 *
 *     q = rint((double)value * inv_p)
 *     r = fma(neg_p, q, (double)value)
 *
 * A 64-byte SVE vector contains sixteen INT32 lanes but only eight FP64
 * lanes.  The widening conversion maps d[i] to s[2*i], so the implementation
 * processes even and odd INT32 lanes as two FP64 streams, then narrows and
 * ZIP1-interleaves them back into the original INT32 lane order.
 *
 * In the fused NN build, the existing kernel C/ldc argument slots are kept as
 * a virtual INT32 coordinate plane so the proven GEMM traversal is unchanged.
 * The numeric base is the same as `c8`; the virtual plane is never dereferenced.
 * Converting the virtual byte displacement to an INT32 element displacement
 * therefore gives the real C8 byte address directly, without loading `ldc8`.
 *
 * x4/x5/x6 are dead after GEMM reaches SAVE_ZACOL; x7 remains reserved for the kernel C slot.  p5/p6 are temporary
 * predicates: MOPA predicates for the just-finished M block, rebuilt before a
 * later M block can accumulate.  The live p0-p4/p7 traversal predicates and
 * the p1/p2 MOVA predicates are left intact.  z16-z22 are epilogue-only
 * temporaries in the production (non-DOUBLE) INT8 save schedule; the current
 * value and the two macro-provided scratch registers are never overwritten.
 */
.macro INT8_INVERSE_SCALING_DIRECT pc,off,reg_pg,value,tmp_q,tmp_mod
    // The 11th argument (buf) is at incoming sp+8; SAVE_REGS subtracts 176.
    ldr     x4, [sp, #(11 * 16 + 8)]
    cbz     x4, 9f

    // Read the per-call constants before x4 is reused for the destination.
    ldr     w6, [x4, #INVERSE_SCALING_PARAMS_MODULUS]
    ldr     x5, [x4, #INVERSE_SCALING_PARAMS_C8]
    cbz     w6, 8f

    // The constants are prepared once by the C++ caller.  Broadcast them to
    // FP64 lanes; no SDIV or reciprocal operation is performed in the kernel.
    ldr     x6, [x4, #INVERSE_SCALING_PARAMS_INV_P]
    dup     z16.d, x6
    ldr     x6, [x4, #INVERSE_SCALING_PARAMS_NEG_P]
    dup     z17.d, x6

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

    // A widening SVE .s -> .d conversion maps destination d[i] to source
    // s[2*i], not to source s[i].  Split the source into its *even* and *odd*
    // INT32 lanes, rather than into low/high contiguous halves.  x6 holds the
    // number of even lanes (ceil(count/2)); x5 then holds odd lanes
    // (floor(count/2)).
    cntp    x5, \reg_pg, \reg_pg\().s
    add     x6, x5, #1
    lsr     x6, x6, #1
    sub     x5, x5, x6
    whilelt p5.d, xzr, x6
    whilelt p6.d, xzr, x5

    // Convert values {0,2,4,...} and {1,3,5,...} independently.  Rotating by
    // one INT32 lane makes the original odd lanes occupy source positions
    // {0,2,4,...} for the second widening conversion.  SVE EXT is destructive,
    // so z18 must first receive a copy of the source vector.
    mov     z18.d, \value\().d
    ext     z18.b, z18.b, z18.b, #4
    scvtf   z19.d, p5/m, \value\().s
    scvtf   z20.d, p6/m, z18.s

    // q = rint(value * inv_p).  Keep the converted original value in z19/z20
    // because the following FMLA must compute value + q * (-p), not q +
    // q * (-p).
    mov     z21.d, z19.d
    fmul    z21.d, p5/m, z21.d, z16.d
    mov     z22.d, z20.d
    fmul    z22.d, p6/m, z22.d, z16.d
    frintn  z21.d, p5/m, z21.d
    frintn  z22.d, p6/m, z22.d
    fmla    z19.d, p5/m, z21.d, z17.d
    fmla    z20.d, p6/m, z22.d, z17.d

    // The remainder is integral and within INT8 for every supported modulus.
    // First compact each FP64 result vector to bytes; ZIP1 then restores the
    // original lane order: even0, odd0, even1, odd1, ... .
    fcvtzs  z21.d, p5/m, z19.d
    fcvtzs  z22.d, p6/m, z20.d
    uzp1    z21.s, z21.s, z21.s
    uzp1    z21.h, z21.h, z21.h
    uzp1    z21.b, z21.b, z21.b
    uzp1    z22.s, z22.s, z22.s
    uzp1    z22.h, z22.h, z22.h
    uzp1    z22.b, z22.b, z22.b
    zip1    z18.b, z21.b, z22.b

    // Recover the original active-lane count and write one contiguous C8
    // vector.  The byte predicate preserves partial final M tiles.
    add     x5, x5, x6
    whilelt p5.b, xzr, x5
    st1b    { z18.b }, p5, [x4]
    b       9f

8:
    // Mode zero is the established low-byte path and must not execute any
    // floating-point or division instruction.
    addvl   x4, \pc, \off
    sub     x4, x4, x5
    lsr     x4, x4, #2
    add     x4, x5, x4
    cntp    x5, \reg_pg, \reg_pg\().s
    whilelt p6.b, xzr, x5
    uzp1    \value\().h, \value\().h, \value\().h
    uzp1    \value\().b, \value\().b, \value\().b
    st1b    { \value\().b }, p6, [x4]
9:
.endm
