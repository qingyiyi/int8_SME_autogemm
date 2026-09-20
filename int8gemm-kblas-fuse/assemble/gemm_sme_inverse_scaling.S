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
 * The conversion is deliberately performed in two halves.  A 64-byte SVE
 * vector contains sixteen INT32 lanes but only eight FP64 lanes; a widening
 * SCVTF/FCVTZS operation consumes/produces one half at a time.  The low and
 * high results are narrowed and stored separately, preserving INT32 lane order.
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

    // Count active INT32 lanes.  Split that count into the low/high FP64
    // halves.  x5 is low_count (0..8), x6 is high_count (0..8).
    cntp    x5, \reg_pg, \reg_pg\().s
    mov     x6, x5
    subs    x6, x6, #8
    csel    x6, x6, xzr, hs
    sub     x5, x5, x6
    whilelt p5.d, xzr, x5
    whilelt p6.d, xzr, x6

    // SCVTF with a .d destination consumes the lower half of the .s source.
    // EXT exposes the original INT32 high half in that same lower position.
    ext     z18.b, \value\().b, \value\().b, #32
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
    // Keep each half as FP64 until conversion, then narrow to one byte per
    // original lane (d -> s -> h -> b).
    fcvtzs  z21.d, p5/m, z19.d
    fcvtzs  z22.d, p6/m, z20.d
    uzp1    z21.s, z21.s, z21.s
    uzp1    z21.h, z21.h, z21.h
    uzp1    z21.b, z21.b, z21.b
    uzp1    z22.s, z22.s, z22.s
    uzp1    z22.h, z22.h, z22.h
    uzp1    z22.b, z22.b, z22.b

    // Store low and high halves consecutively.  Rebuild byte predicates after
    // the FP64 work; each predicate now describes bytes, not FP64 lanes.
    whilelt  p5.b, xzr, x5
    whilelt  p6.b, xzr, x6
    st1b    { z21.b }, p5, [x4]
    add     x6, x4, #8
    st1b    { z22.b }, p6, [x6]
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
