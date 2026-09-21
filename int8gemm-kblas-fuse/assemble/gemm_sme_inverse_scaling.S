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
 * The parameter block is invariant during one kernel invocation, but the GEMM
 * compute phase needs the ordinary working registers.  PREPARE therefore runs
 * once at the entry of each completed ZA save tile, after accumulation has
 * finished.  It caches:
 *
 *   x5      C8 tile base and null-store sentinel
 *   w6      one-bit mode state (zero: low-byte mode; one: FP64 inverse path)
 *   z27.d   broadcast inv_p
 *   z28.d   broadcast neg_p
 *
 * The scalar modulus is loaded exactly once by PREPARE.  DIRECT tests the
 * cached one-bit state before reusing x6 for lane counts, then restores w6
 * before returning so the next vector in the same save tile sees the same
 * mode.  The virtual C32 base/stride registers are part of the proven
 * traversal and must never be used as epilogue scratch.  p5/p6 are temporary
 * even/odd FP64 predicates; p5 is reused as the final byte-store predicate
 * after the FP64 work finishes.
 * The live p0-p4/p7 traversal predicates and p1/p2 MOVA predicates are left
 * intact.  z18-z22 are epilogue-only temporaries in the production
 * (non-DOUBLE) save schedule; z27/z28 are not used by any save schedule after
 * PREPARE.
 */

/*
 * Load the invariant fused-store state once for a completed ZA save tile.
 *
 * The 11th argument (buf) is at incoming sp+8; SAVE_REGS subtracts 176 bytes.
 * x5 and w6 are cleared first so a null buf, a null c8 base, or mode zero
 * cannot retain state from a preceding tile.  Mode zero intentionally avoids
 * both FP64 constant loads and FP64 arithmetic in the direct helper.
 *
 * The target BiSheng assembler permits only p0-p7 in SVE memory operations.
 * Keeping the mode in w6 avoids the former high-predicate dependency entirely
 * while retaining the tile-level preload of c8, inv_p, and neg_p.
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

    ldr     x6, [x4, #INVERSE_SCALING_PARAMS_INV_P]
    dup     z27.d, x6
    ldr     x6, [x4, #INVERSE_SCALING_PARAMS_NEG_P]
    dup     z28.d, x6

    // Only the mode is needed after the broadcasts.  Canonicalize it so the
    // direct helper can restore it after using x6 as a scalar scratch.
    mov     w6, #1
9:
.endm

/*
 * Consume one MOVA result vector using state prepared by
 * INT8_INVERSE_SCALING_PREPARE.  There are no parameter-block loads or
 * constant broadcasts in this hot per-vector path.
 */
.macro INT8_INVERSE_SCALING_DIRECT pc,off,reg_pg,value,tmp_q,tmp_mod
    // x5 is a stable C8 base for the whole save tile; PREPARE clears it for a
    // null parameter block, so it also safely suppresses stores in that case.
    cbz     x5, 9f

    // w6 was loaded once by PREPARE as a mode bit.  It is restored at the end
    // of each path because x6 is otherwise the per-vector lane-count scratch.
    cbz     w6, 8f

    // A widening SVE .s -> .d conversion maps destination d[i] to source
    // s[2*i], not to source s[i].  Split the source into its even and odd
    // INT32 lanes.  x4 holds ceil(count/2); x6 then holds floor(count/2).
    cntp    x6, \reg_pg, \reg_pg\().s
    add     x4, x6, #1
    lsr     x4, x4, #1
    sub     x6, x6, x4
    whilelt p5.d, xzr, x4
    whilelt p6.d, xzr, x6

    // The proven GEMM traversal presents a virtual INT32 address plane whose
    // numeric base is c8_tile.  Converting its byte displacement to an
    // element displacement maps it directly onto the byte-addressed C8 tile:
    //
    //   dst = c8_tile + ((pc + off * VL) - c8_tile) / sizeof(int32_t)
    addvl   x4, \pc, \off
    sub     x4, x4, x5
    lsr     x4, x4, #2
    add     x4, x5, x4

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
    // q * (-p).  z27/z28 were broadcast once by PREPARE.
    mov     z21.d, z19.d
    fmul    z21.d, p5/m, z21.d, z27.d
    mov     z22.d, z20.d
    fmul    z22.d, p6/m, z22.d, z27.d
    frintn  z21.d, p5/m, z21.d
    frintn  z22.d, p6/m, z22.d
    fmla    z19.d, p5/m, z21.d, z28.d
    fmla    z20.d, p6/m, z22.d, z28.d

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

    // Rebuild the C8 predicate after p5 has been used for FP64 lanes.  Doing
    // CNT P again avoids retaining the count in the live virtual-stride
    // register.
    cntp    x6, \reg_pg, \reg_pg\().s
    whilelt p5.b, xzr, x6
    st1b    { z18.b }, p5, [x4]
    mov     w6, #1
    b       9f

8:
    // Mode zero is the established low-byte path and must not execute FP64
    // arithmetic.  p5 is a low predicate register accepted by the target
    // assembler for the final byte store.
    cntp    x6, \reg_pg, \reg_pg\().s
    addvl   x4, \pc, \off
    sub     x4, x4, x5
    lsr     x4, x4, #2
    add     x4, x5, x4
    whilelt p5.b, xzr, x6
    uzp1    \value\().h, \value\().h, \value\().h
    uzp1    \value\().b, \value\().b, \value\().b
    st1b    { \value\().b }, p5, [x4]
    mov     w6, wzr
9:
.endm
