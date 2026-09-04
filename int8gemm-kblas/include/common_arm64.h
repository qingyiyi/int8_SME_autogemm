/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: arm64 embedded assembly
 * Author: KPL
 * Create: 2020-11-05
 * Notes: NA
 */

#ifndef COMMON_ARM64
#define COMMON_ARM64

#include "blas_types_def.h"

#define MB __asm__ __volatile__("dmb  ish" : : : "memory")
#define WMB __asm__ __volatile__("dmb  ishst" : : : "memory")
#define RMB __asm__ __volatile__("dmb  ishld" : : : "memory")

#ifndef F_INTERFACE_FLANG
#define RETURN_BY_COMPLEX
#else
#define RETURN_BY_STACK
#endif

#if !defined(ASSEMBLER)
static inline void BlasLock(volatile const BLASULONG *address)
{
    BLASULONG ret;
    do {
        while (*address) {
            YIELDING;
        }

        __asm__ __volatile__("mov x4, #1                                   \n\t"
            "sevl                                                          \n\t"
            "1:                                                            \n\t"
            "wfe                                                           \n\t"
            "2:                                                            \n\t"
            "ldaxr x2, [%1]                                                \n\t"
            "cbnz  x2, 1b                                                  \n\t"
            "stxr  w3, x4, [%1]                                            \n\t"
            "cbnz  w3, 2b                                                  \n\t"
            "mov   %0, #0                                                  \n\t"
            : "=r"(ret), "=r"(address)
            : "1"(address)
            : "memory", "x2", "x3", "x4"
        );
    } while (ret);
}

#define BLAS_LOCK_DEFINED

static inline unsigned long long rpcc(void)
{
    unsigned long long ret = 0;
    BLASULONG shift;

    __asm__ __volatile__("isb; mrs %0,cntvct_el0" : "=r"(ret));
    __asm__ __volatile__("mrs %0,cntfrq_el0; clz %w0, %w0" : "=&r"(shift));

    return ret << shift;
}

#define RPCC_DEFINED
#define RPCC64BIT

static inline BLASULONG BlasQuickdivide(BLASULONG x, BLASULONG y)
{
    return x / y;
}

#ifndef DOUBLE
#define GET_IMAGE(res) __asm__ __volatile__("str s1, %0" : "=m"(res) : : "memory")
#else
#define GET_IMAGE(res) __asm__ __volatile__("str d1, %0" : "=m"(res) : : "memory")
#endif

#define GET_IMAGE_CANCEL

#endif

#ifdef F_INTERFACE
#define REALNAME ASMFNAME
#else
#define REALNAME ASMNAME
#endif

#if defined(ASSEMBLER) && !defined(NEEDPARAM)

.macro PROLOGUE
    .text;
    .p2align 2;
    .global REALNAME;
    .hidden REALNAME;
    .type REALNAME, % function;
REALNAME:
.endm

.macro SAVE_REGS
    .align 5
    add     sp, sp, #-(11 * 16)
    stp     d8, d9, [sp, #(0 * 16)]     // SAVE_REGS
    stp     d10, d11, [sp, #(1 * 16)]
    stp     d12, d13, [sp, #(2 * 16)]
    stp     d14, d15, [sp, #(3 * 16)]
    stp     d16, d17, [sp, #(4 * 16)]   // SAVE_REGS
    stp     x18, x19, [sp, #(5 * 16)]
    stp     x20, x21, [sp, #(6 * 16)]
    stp     x22, x23, [sp, #(7 * 16)]
    stp     x24, x25, [sp, #(8 * 16)]   // SAVE_REGS
    stp     x26, x27, [sp, #(9 * 16)]
    stp     x28, x29, [sp, #(10 * 16)]  // SAVE_REGS
.endm

.macro RESTORE_REGS
    ldp     d8, d9, [sp, #(0 * 16)]     // RESTORE_REGS
    ldp     d10, d11, [sp, #(1 * 16)]
    ldp     d12, d13, [sp, #(2 * 16)]
    ldp     d14, d15, [sp, #(3 * 16)]
    ldp     d16, d17, [sp, #(4 * 16)]   // RESTORE_REGS
    ldp     x18, x19, [sp, #(5 * 16)]
    ldp     x20, x21, [sp, #(6 * 16)]
    ldp     x22, x23, [sp, #(7 * 16)]
    ldp     x24, x25, [sp, #(8 * 16)]   // RESTORE_REGS
    ldp     x26, x27, [sp, #(9 * 16)]
    ldp     x28, x29, [sp, #(10 * 16)]  // RESTORE_REGS
    add     sp, sp, #(11*16)
.endm

.macro START_SME_FEATURE
    fmov x23, d0
    fmov x24, d1
    fmov x25, d2
    fmov x26, d3
    msr SVCRSMZA, #1
    isb
    fmov d0, x23
    fmov d1, x24
    fmov d2, x25
    fmov d3, x26
.endm

.macro STOP_SME_FEATURE
    msr SVCRSMZA, #0
    isb
.endm

#define EPILOGUE

#define PROFCODE

#endif

#ifndef PAGESIZE
#define PAGESIZE (4 << 10)
#endif
#define HUGE_PAGESIZE (4 << 20)
#ifndef EXT_BUFFER_SIZE
#define EXT_BUFFER_SIZE (4 << 20)
#endif
#ifndef BUFFERSIZE
#if defined(CORTEXA57)
#define BUFFER_SIZE (20 << 20)
#else
#define BUFFER_SIZE (36 << 20)
#endif
#else
#define BUFFER_SIZE (32 << BUFFERSIZE)
#endif

#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif
#ifndef GET_BUF_SIZE
#define GET_BUF_SIZE(sz, TYPE) (sizeof(TYPE) * (sz) * COMPSIZE)
#endif
#endif