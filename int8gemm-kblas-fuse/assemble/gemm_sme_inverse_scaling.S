/*
 * Production INT8 inverse-scaling epilogue.
 *
 * The NN GEMM kernel first completes its proven SME/ZA INT32 store.  Before
 * returning to the driver, this epilogue reads the tile, applies the exact
 * centered-remainder semantics of the original C++ implementation, and writes
 * the corresponding INT8 tile.  The GEMM accumulation and packing code remain
 * unchanged; only the inverse-scaling operation has moved into the kernel.
 *
 * Int8FusedStoreParams layout (bytes):
 *   0  c32       8  c8       16 ldc32       24 ldc8
 *   32 rows      40 cols    48 modulus     52 reserved
 */

#define INVERSE_SCALING_PARAMS_C32      0
#define INVERSE_SCALING_PARAMS_C8       8
#define INVERSE_SCALING_PARAMS_LDC32    16
#define INVERSE_SCALING_PARAMS_LDC8     24
#define INVERSE_SCALING_PARAMS_ROWS     32
#define INVERSE_SCALING_PARAMS_COLS     40
#define INVERSE_SCALING_PARAMS_MODULUS  48

.macro INT8_INVERSE_SCALING_POSTSTORE
    // The existing prologue reserves 11*16 bytes.  The original ninth and
    // tenth GP arguments (ldc and buf) are therefore at sp+176 and sp+184.
    ldr     x17, [sp, #(11 * 16 + 8)]
    cbz     x17, .L_inverse_scaling_store_done

    ldr     x9,  [x17, #INVERSE_SCALING_PARAMS_C32]
    ldr     x10, [x17, #INVERSE_SCALING_PARAMS_C8]
    ldr     x11, [x17, #INVERSE_SCALING_PARAMS_LDC32]
    ldr     x12, [x17, #INVERSE_SCALING_PARAMS_LDC8]
    ldr     x13, [x17, #INVERSE_SCALING_PARAMS_ROWS]
    ldr     x14, [x17, #INVERSE_SCALING_PARAMS_COLS]
    ldr     w15, [x17, #INVERSE_SCALING_PARAMS_MODULUS]

    cbz     x13, .L_inverse_scaling_store_done
    cbz     x14, .L_inverse_scaling_store_done

    // stnt1w is the producer of c32.  Order it before the scalar loads below.
    dmb     ish

    // mode == 0 is the low-byte operation.  strb intentionally stores the
    // low eight bits, including for negative INT32 values.
    cbz     w15, .L_inverse_scaling_store_low_byte

    // All supported moduli are positive odd integers.  For an integer input,
    // truncating signed division followed by centered remainder correction is
    // exactly round-to-nearest quotient semantics (there are no half ties).
    lsr     w7, w15, #1
    mov     x0, x14
.L_inverse_scaling_store_mod_col:
    mov     x1, x13
    mov     x2, x9
    mov     x3, x10
.L_inverse_scaling_store_mod_row:
    ldr     w4, [x2], #4
    sdiv    w5, w4, w15
    msub    w6, w5, w15, w4
    cmp     w6, w7
    ble     .L_inverse_scaling_store_mod_check_low
    sub     w6, w6, w15
    b       .L_inverse_scaling_store_mod_write
.L_inverse_scaling_store_mod_check_low:
    // If remainder + floor(p/2) is negative, remainder < -floor(p/2).
    cmn     w6, w7
    bge     .L_inverse_scaling_store_mod_write
    add     w6, w6, w15
.L_inverse_scaling_store_mod_write:
    strb    w6, [x3], #1
    subs    x1, x1, #1
    b.ne    .L_inverse_scaling_store_mod_row

    add     x9, x9, x11, lsl #2
    add     x10, x10, x12
    subs    x0, x0, #1
    b.ne    .L_inverse_scaling_store_mod_col
    b       .L_inverse_scaling_store_done_barrier

.L_inverse_scaling_store_low_byte:
    mov     x0, x14
.L_inverse_scaling_store_low_col:
    mov     x1, x13
    mov     x2, x9
    mov     x3, x10
.L_inverse_scaling_store_low_row:
    ldr     w4, [x2], #4
    strb    w4, [x3], #1
    subs    x1, x1, #1
    b.ne    .L_inverse_scaling_store_low_row

    add     x9, x9, x11, lsl #2
    add     x10, x10, x12
    subs    x0, x0, #1
    b.ne    .L_inverse_scaling_store_low_col

.L_inverse_scaling_store_done_barrier:
    // Make the newly written INT8 tile visible before returning to the driver.
    dmb     ish
.L_inverse_scaling_store_done:
.endm
