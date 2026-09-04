/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2020-2020. All rights reserved.
 * Description: linux parameters and functions
 * Author: KPL
 * Create: 2020-11-05
 * Notes: NA
 */

#ifndef COMMON_LINUX_H
#define COMMON_LINUX_H

#ifndef ASSEMBLER

#include <syscall.h>
#include <unistd.h>

#if !defined(MPOL_PREFERRED)
#define MPOL_PREFERRED 1
#endif

#if !defined(MPOL_INTERLEAVE)
#define MPOL_INTERLEAVE 3
#endif

static inline int UniMbind(const void *addr, unsigned long len, int mode, const unsigned long *nodeMask,
    unsigned long maxNode, unsigned flags)
{
    return syscall(SYS_mbind, addr, len, mode, nodeMask, maxNode, flags);
}

static inline int UniSetMempolicy(int mode, const unsigned long *addr, unsigned long flag)
{
    return syscall(SYS_set_mempolicy, mode, addr, flag);
}

static inline int UniGettid(void)
{
#ifndef SYS_gettid
    return getpid();
#else
    return syscall(SYS_gettid);
#endif
}

#endif
#endif