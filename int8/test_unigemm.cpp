#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <hbwmalloc.h>
#include <sched.h>
#include <dlfcn.h>
#include <numa.h>
#include <numaif.h>
#include <sys/mman.h>
#include "unigemm_sme.h"

extern "C" {
#include "kblas.h"
#include "int8_gemm.h"
}

#define HBM_ALIGNED_SIZE (1 << 21)

int get_hbm_node() {
    int cpu = sched_getcpu();
    return (cpu / 38) + 16;
}
size_t round_up_2mb(size_t size) {
    return (size + 0x1FFFFF) & ~0x1FFFFF;
}
void* aligned_alloc_generic(size_t alignment, size_t size) {
#ifdef __linux__
    if (1) {
        size_t huge_size = round_up_2mb(size);
        void* p = mmap(nullptr, huge_size, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);

        if (p == MAP_FAILED) {
            fprintf(stderr, "huge alloc of %zu bytes failed\n", huge_size);
            abort();
        }
#if __has_include(<numaif.h>)
        unsigned long nodemask = 1UL << get_hbm_node();
        mbind(p, huge_size, MPOL_BIND, &nodemask, sizeof(nodemask) * 8, MPOL_MF_STRICT);
#endif
        return p;
    }
#endif
    if (1) {
    }
    void* p = nullptr;
    posix_memalign(&p, alignment, size);
    return p;
}

void aligned_free_generic(void* p, size_t alloc_size) {
#ifdef __linux__
    if (1) {
        munmap(p, round_up_2mb(alloc_size));
        return;
    }
#endif
}

int malloc_on_hbm_with_check( void **addr, size_t length_bytes, const char *buffer_name)
{
   void *mapAddress = NULL;
   mapAddress = mmap(NULL, length_bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
   if( mapAddress == MAP_FAILED ){
      printf( "mmap failed for %s. Skip.\n", buffer_name );
      exit(-1);
   }
   int cpuid = sched_getcpu();
   unsigned long nodeid = numa_node_of_cpu(cpuid);
   int numaNodes = numa_num_configured_nodes();
   int hbmNodeOffset = numaNodes / 2;
   unsigned long hbmNode = nodeid + hbmNodeOffset;
   nodemask_t nodemask;
   struct bitmask bitmask = {NUMA_NUM_NODES, nodemask.n};

   numa_bitmask_clearall(&bitmask);
   numa_bitmask_setbit(&bitmask, hbmNode);    
   int ret = mbind(mapAddress, length_bytes, MPOL_BIND, nodemask.n, NUMA_NUM_NODES, 0);

   if (ret != 0) {
      printf( "mbind failed for %s. Skip.\n", buffer_name );
      exit(-1);
   }
   if (hbw_verify_memory_region(mapAddress, length_bytes, HBW_TOUCH_PAGES) != 0) {
      printf( "Buffer %s is not allocated in HBM. Skip.\n", buffer_name );
      exit(-1);
   }
   *addr = mapAddress;
   return 0;
}

struct TestStats {
    int total = 0;
    int passed = 0;
    int failed = 0;
    size_t m;
    size_t n;
    size_t k;
    void pass() { total++; passed++; }
    void fail() { total++; failed++; }
    void print() const {
        printf("  %d/%d passed", passed, total);
        if (failed) printf(" (%d FAILED)", failed);
        printf("\n");
    }
};


inline double int8_kblas_notrans_blas(       // alpha 1   beta 0
    size_t m, size_t n, size_t k,
    const int8_t *A8i, size_t lda8i,
    const int8_t *B8i, size_t ldb8i,
    int32_t *Ctmp, size_t ldc32i
)
{
    /************* temp ***********************/
    int8_t *sa = NULL;
    int ret = malloc_on_hbm_with_check( (void **)(&sa), 2048*128*sizeof(int8_t)*2*32*2, "sa" );
    if (ret != 0) {
        printf("malloc_on_hbm failed, Out of memory for sa\n");
        exit(1);
    }
    int8_t *sb = NULL; 
    ret = malloc_on_hbm_with_check( (void **)(&sb), 2048*8192*sizeof(int8_t)*2*2, "sb" );
    if (ret != 0) {
        printf("malloc_on_hbm failed, Out of memory for sb\n");
        exit(1);
    }
    /************* temp ***********************/


    // Initialize output to zero
    const size_t c_size = ldc32i * n;
    std::memset(Ctmp, 0, c_size * sizeof(int32_t));

    auto start_inner_int8_gemm = std::chrono::high_resolution_clock::now();

    CBLAS_LAYOUT layout = CblasColMajor;
    CBLAS_TRANSPOSE transa = CblasNoTrans;
    CBLAS_TRANSPOSE transb = CblasNoTrans;
    CBLAS_OFFSET offsetc = CblasFixOffset;
    int32_t oc_ = 0;
    int32_t *oc = &oc_;

    

    // printf("offsetc = %d\n", offsetc);
    cblas_gemm_s8s8s32(layout, transa, transb, offsetc, m, n, k, 1, A8i, lda8i, 0, B8i, ldb8i, 0, 0, Ctmp, ldc32i, oc, sa, sb);
    // printf("ctmp[0] = %d\n", Ctmp[0]);

    auto end_inner_int8_gemm = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inner_int8_gemm_time = end_inner_int8_gemm - start_inner_int8_gemm;
    
    /*************temp****************/
    if (munmap(sa, 2048*256*sizeof(int8_t)*2*32) == -1) {
        printf( "Error while unmapping sa\n");
    }
    if (munmap(sb, 2048*8192*sizeof(int8_t)*2*2) == -1) {
        printf( "Error while unmapping sb\n");
    }
    /*************temp****************/
    double gflops = 2*m*n*k/1e9/inner_int8_gemm_time.count();
    double effi = gflops / (1.55 * 128 * 32 * 16) * 100;
    std::cout << "M=" << m << " N=" << n << " K=" << k << "ldc=" << ldc32i << "  int8_gemm_time: " << std::fixed << std::setprecision(6) << inner_int8_gemm_time.count() << " seconds"
            << " performance: " << gflops << " GFLOPs" << " effi : " << effi << "%" << std::endl;
    return inner_int8_gemm_time.count();
}



inline double int8_gemm_notrans_blas(       // alpha 1   beta 0
    size_t m, size_t n, size_t k,
    const int8_t *A8i, size_t lda8i,
    const int8_t *B8i, size_t ldb8i,
    int32_t *Ctmp, size_t ldc32i
)
{
    // Initialize output to zero
    const size_t c_size = ldc32i * n;
    std::memset(Ctmp, 0, c_size * sizeof(int32_t));

    auto start_inner_int8_gemm = std::chrono::high_resolution_clock::now();

    unigemm_sme_int8_mt(B8i, ldb8i, A8i, lda8i, Ctmp, ldc32i, n, m, k);

    auto end_inner_int8_gemm = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> inner_int8_gemm_time = end_inner_int8_gemm - start_inner_int8_gemm;
    double gflops = 2*m*n*k/1e9/inner_int8_gemm_time.count();
    double effi = gflops / (1.55 * 128 * 32 * 16) * 100;
    std::cout << "M=" << m << " N=" << n << " K=" << k << "ldc=" << ldc32i << "  int8_gemm_time: " << std::fixed << std::setprecision(6) << inner_int8_gemm_time.count() << " seconds"
            << " performance: " << gflops << " GFLOPs" << " effi : " << effi << "%" << std::endl;
    return inner_int8_gemm_time.count();
}

void naive_notrans(
    size_t m, size_t n, size_t k,
    const int8_t *A8i, size_t lda8i,
    const int8_t *B8i, size_t ldb8i,
    int32_t *Cref, size_t ldc32i)
{
    std::memset(Cref, 0, ldc32i * n * sizeof(int32_t));
    for (size_t col = 0; col < n; ++col)
        for (size_t row = 0; row < m; ++row)
            for (size_t inner = 0; inner < k; ++inner)
                Cref[col * ldc32i + row] +=
                    static_cast<int32_t>(A8i[inner * lda8i + row]) *
                    static_cast<int32_t>(B8i[col * ldb8i + inner]);
}

void fill_A_notrans(int8_t *A8i, size_t m, size_t k, size_t lda, unsigned seed) {
    std::srand(seed);
    for (size_t inner = 0; inner < k; ++inner)
        for (size_t row = 0; row < m; ++row)
            A8i[inner * lda + row] = static_cast<int8_t>((int(std::rand()) % 256) - 128);
}

void fill_B_notrans(int8_t *B8i, size_t k, size_t n, size_t ldb, unsigned seed) {
    std::srand(seed);
    for (size_t col = 0; col < n; ++col)
        for (size_t inner = 0; inner < k; ++inner)
            B8i[col * ldb + inner] = static_cast<int8_t>((int(std::rand()) % 256) - 128);
}

bool compare(const int32_t *a, const int32_t *b,
             size_t m, size_t n, size_t ldc,
             const char *label)
{
    for (size_t col = 0; col < n; ++col) {
        for (size_t row = 0; row < m; ++row) {
            size_t idx = col * ldc + row;
            if (a[idx] != b[idx]) {
                printf("    FAIL [%s]: C[%zu][%zu]=%d, expected %d\n",
                       label, row, col, a[idx], b[idx]);
                return false;
            }
            else if(col < 4 && row < 4){
                printf("Result %d %d: %d %d\n", col, row, a[idx], b[idx]);
            }
        }
    }
    return true;
}

bool test_square(TestStats& s, const char* mode, bool verify) {
    const size_t M = s.m;
    const size_t N = s.n;
    const size_t K = s.k;

    const size_t lda = M;
    const size_t ldb = K;
    const size_t ldc = (M * 4 + 64) / 4;



    size_t num_moduli = 10;

    if(verify){
        num_moduli = 1;
    }
    int8_t *A8i = NULL;
    int8_t *B8i = NULL;
    int32_t *Ctest = NULL;

    size_t sizeA = M * K * sizeof(int8_t) * num_moduli;
    size_t sizeB = K * N * sizeof(int8_t) * num_moduli;
    size_t sizeC = ldc * N * sizeof(int32_t) * num_moduli;

    sizeA = (sizeA + HBM_ALIGNED_SIZE - 1) / HBM_ALIGNED_SIZE * HBM_ALIGNED_SIZE;
    sizeB = (sizeB + HBM_ALIGNED_SIZE - 1) / HBM_ALIGNED_SIZE * HBM_ALIGNED_SIZE;
    sizeC = (sizeC + HBM_ALIGNED_SIZE - 1) / HBM_ALIGNED_SIZE * HBM_ALIGNED_SIZE;

    printf("sizeA=%zu, sizeB=%zu, sizeC=%zu\n", sizeA, sizeB, sizeC);

    int ret;
    ret = malloc_on_hbm_with_check( (void **)(&A8i), sizeA, "A8i" );
    if (ret != 0) {
        printf("malloc_on_hbm failed, Out of memory for A8i\n");
        exit(1);
    }

    ret = malloc_on_hbm_with_check( (void **)(&B8i), sizeB, "B8i" );
    if (ret != 0) {
        printf("malloc_on_hbm failed, Out of memory for B8i\n");
        exit(1);
    }

    ret = malloc_on_hbm_with_check( (void **)(&Ctest), sizeC, "Ctest" );
    if (ret != 0) {
        printf("malloc_on_hbm failed, Out of memory for Ctest\n");
        exit(1);
    }

    // int32_t *Cref = NULL;
    // size_t sizeCref = ldc * N * sizeof(int32_t);
    // Cref = (int32_t*)aligned_alloc_generic(64, sizeCref);
    // if (Cref == NULL) {
    //     printf("aligned_alloc failed for Cref\n");
    //     exit(1);
    // }

    // for( int i=0; i<num_moduli; i++) {
    //     int8_t *A8i_tmp = A8i + i * M * K;
    //     int8_t *B8i_tmp = B8i + i * K * N;
    //     int32_t *Ctest_tmp = Ctest + i * ldc * N;

    //     fill_A_notrans(A8i_tmp, M, K, lda, 42 + i);   // 不同 seed 产生不同数据
    //     fill_B_notrans(B8i_tmp, K, N, ldb, 123 + i);
        
    //     // 自研KBLAS
    //     int8_kblas_notrans_blas(M, N, K, A8i_tmp, lda, B8i_tmp, ldb, Ctest_tmp, ldc);
        
    //     // naive_notrans(M, N, K, A8i_tmp, lda, B8i_tmp, ldb, Cref, ldc);
    //     // if (compare(Ctest_tmp, Cref, M, N, ldc, "KBLAS")) {
    //     //     s.pass();
    //     // } else {
    //     //     s.fail();
    //     // }

    //     //uigemm
    //     // int8_gemm_notrans_blas(M, N, K, A8i_tmp, lda, B8i_tmp, ldb, Ctest_tmp, ldc);
    //     // if (compare(Ctest_tmp, Cref, M, N, ldc, "unigemm_sme")) {
    //     //     s.pass();
    //     // } else {
    //     //     s.fail();
    //     // }

    // }

    int32_t *Cref = NULL;
    if (verify) {
        size_t sizeCref = ldc * N * sizeof(int32_t);
        Cref = (int32_t*)aligned_alloc_generic(64, sizeCref);
        if (Cref == NULL) { printf("aligned_alloc failed for Cref\n"); exit(1); }
    }

    double kblas_time = 0;
    double unigemm_time = 0;
    
    for( int i=0; i<num_moduli; i++) {
        int8_t *A8i_tmp = A8i + i * M * K;
        int8_t *B8i_tmp = B8i + i * K * N;
        int32_t *Ctest_tmp = Ctest + i * ldc * N;

        fill_A_notrans(A8i_tmp, M, K, lda, 42 + i);
        fill_B_notrans(B8i_tmp, K, N, ldb, 123 + i);

        if (verify) {
            naive_notrans(M, N, K, A8i_tmp, lda, B8i_tmp, ldb, Cref, ldc);
        }

        double temp_time;
        // KBLAS
        if (strcmp(mode, "kblas") == 0 || strcmp(mode, "both") == 0) {
            temp_time = int8_kblas_notrans_blas(M, N, K, A8i_tmp, lda, B8i_tmp, ldb, Ctest_tmp, ldc);
            if (verify) {
                if (compare(Ctest_tmp, Cref, M, N, ldc, "KBLAS")) s.pass(); else s.fail();
            }
            if(i!=0)
                kblas_time+=temp_time;
        }

        // unigemm_sme
        if (strcmp(mode, "unigemm") == 0 || strcmp(mode, "both") == 0) {
            temp_time = int8_gemm_notrans_blas(M, N, K, A8i_tmp, lda, B8i_tmp, ldb, Ctest_tmp, ldc);
            if (verify) {
                if (compare(Ctest_tmp, Cref, M, N, ldc, "unigemm_sme")) s.pass(); else s.fail();
            }
            if(i!=0)
                unigemm_time+=temp_time;
        }
    }

    // average eff print
    if (strcmp(mode, "unigemm") == 0 || strcmp(mode, "both") == 0) {
        double average_unigemm_time = unigemm_time/(num_moduli-1);
        double average_unigemm_effi = (2*M*N*K/1e9/average_unigemm_time)/ (1.55 * 128 * 32 * 16) * 100;
        printf("average unigemm time = %.6lf, average unigemm effi = %.2lf\n", average_unigemm_time, average_unigemm_effi);
    }
    if (strcmp(mode, "kblas") == 0 || strcmp(mode, "both") == 0) {
        double average_kblas_time = kblas_time/(num_moduli-1);
        double average_kblas_effi = (2*M*N*K/1e9/average_kblas_time)/ (1.55 * 128 * 32 * 16) * 100;
        printf("average int8gemm time = %.6lf, average int8gemm effi = %.2lf\n", average_kblas_time, average_kblas_effi);
    }


    if (munmap(A8i, sizeA) == -1) {
        printf( "Error while unmapping A8i\n");
    }
    if (munmap(B8i, sizeB) == -1) {
        printf( "Error while unmapping B8i\n");
    }
    if (munmap(Ctest, sizeC) == -1) {
        printf( "Error while unmapping Ctest\n");
    }
    
    if (verify) {
        aligned_free_generic(Cref, ldc * N * sizeof(int32_t));
    }

    // aligned_free_generic(Cref, sizeCref);

    // return ok;
    return 0;
}

int main(int argc, const char **argv) {
    printf("=======================================================\n");
    printf("INT8 GEMM Non-Transpose: Cross-Implementation Tests\n");
    printf("=======================================================\n\n");

    TestStats stats;
    stats.m = atoi(argv[1]);
    stats.n = atoi(argv[2]);
    stats.k = atoi(argv[3]);

    const char* mode   = (argc > 4) ? argv[4] : "both";          // kblas / unigemm / both
    bool        verify = (argc > 5) ? (strcmp(argv[5],"verify")==0 || atoi(argv[5])!=0) : false;

    test_square(stats, mode, verify);


    printf("\n=======================================================\n");
    printf("Summary: "); stats.print();
    printf("=======================================================\n");

    return stats.failed ? 1 : 0;
}