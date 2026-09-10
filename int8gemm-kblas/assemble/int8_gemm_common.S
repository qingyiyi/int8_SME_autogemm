#ifndef GEMM_INT8_COMMON
#define GEMM_INT8_COMMON

#if defined(LHS_INT) && defined(LHS_UINT)
#error "Both LHS_INT and LHS_UINT are defined"
#endif
#if defined(RHS_INT) && defined(RHS_UINT)
#error "Both RHS_INT and RHS_UINT are defined"
#endif

// #if defined (LHS_INT) && defined(RHS_INT)
    #define COMPUTE_MOPA_TYPED smopa
// #elif defined (LHS_UINT) && defined(RHS_UINT)
//     #define COMPUTE_MOPA_TYPED umopa
//#else // (defined (LHS_INT) && defined(RHS_INT)) || (defined (LHS_UINT) && defined(RHS_UINT))
//    #ifdef LHS_INT
//        #define COMPUTE_MOPA_TYPED sumopa
//    #else // LHS_INT
//        #define COMPUTE_MOPA_TYPED usmopa
//    #endif // LHS_INT
//#endif // (defined (LHS_INT) && defined(RHS_INT)) || (defined (LHS_UINT) && defined(RHS_UINT))

.macro RE_CALCULATE_PG_MOPA
    // TMP_PREDICATE and Pg_MOPA_1 use the same predicate register(p5) in usual,
    // so it need to calculate the origin Pg_MOPA_1's value again after use TMP_PREDICATE.
	cntp TMP_OFF, p1, p1.s
	lsl TMP_OFF, TMP_OFF, #2
	MOV TMP_VL, #0
	whilelt Pg_MOPA_1.b, TMP_VL, TMP_OFF
.endm

.macro DEAL_LAST_K curK=TMP_CNT
#if defined(SMALL_SME_GEMM)
    // curK = 1 TMP_PTR_SIN=0x000000ff
    // curK = 2 TMP_PTR_SIN=0x0000ffff
    // curK = 3 TMP_PTR_SIN=0x00ffffff
    mov TMP_PTR, #0xffffffff
    mov TMP_VL, #4
    sub TMP_VL, TMP_VL, \curK               // get (4-curK)
    lsl TMP_VL, TMP_VL, #3                  // get (4-curK)*8
    lsr TMP_PTR, TMP_PTR, TMP_VL            // need logical shift right (4-curK)*8 bit

    dup z29.s, TMP_PTR_SIN
#endif
.endm

#endif