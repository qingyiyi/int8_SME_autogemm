#ifndef CONFIG_H
#define CONFIG_H
#define OS_LINUX 1
#define ARCH_ARM64 1
#define C_GCC 1
#define __64BIT__ 1
#define PTHREAD_CREATE_FUNC pthread_create
#define BUNDERSCORE _
#define NEEDBUNDERSCORE 1
#ifndef ARMV9
#define ARMV9
#endif
#define L1_DATA_SIZE 32768
#define L1_DATA_LINESIZE 64
#define L2_SIZE 786432
#define L2_LINESIZE 64
#define DTB_DEFAULT_ENTRIES 64
#define DTB_SIZE 4096
#define L2_ASSOCIATIVE 12
#define VECTOR_LENGTH 512
#define CORE_ARMV9
#define CHAR_CORENAME "ARMV9"
#define GEMM_MULTITHREAD_THRESHOLD 4
#define CACHE_SET_NUM 1024
#endif /* CONFIG_H */