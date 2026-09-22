#include "int8_gemm.hpp"

// The NN assembly keeps its established two output register slots for
// traversal.  In the fused path they carry the current C8 tile address and its
// byte stride; no C32 pointer arithmetic or C32 store remains in C++.
#define KERNEL_OPERATION_SME(M, N, K, ALPHA, SA, LDA, SB, LDB, C8, LDC8, BUF) \
    int8_sme_gemm_kernel_nn                                                     \
    (M, N, K, SA, LDA, ALPHA[0], (FLOAT *)SB, LDB, C8, LDC8, BUF)

//WARNING: 可以改成16X2
int nthreadsM = 32;
int nthreadsN = 1;

// The assembly epilogue receives fixed constants, not the mode index.  The
// reciprocal and the negative modulus are selected once per public GEMM call
// and copied into each tile's parameter block.  Because this table is
// constexpr, the divisions below are compile-time constant evaluation; there
// is no reciprocal division in the GEMM/tile loops.  Mode zero keeps the
// established low-byte path.  The business contract uses num_moduli in [0, 19].
struct InverseScalingConstants {
    int32_t modulus;
    double inv_p;
    double neg_p;
};

constexpr InverseScalingConstants INVERSE_SCALING_CONSTANTS[20] = {
        {0,   0.0,          0.0},
        {255, 1.0 / 255.0,  -255.0},
        {253, 1.0 / 253.0,  -253.0},
        {251, 1.0 / 251.0,  -251.0},
        {247, 1.0 / 247.0,  -247.0},
        {241, 1.0 / 241.0,  -241.0},
        {239, 1.0 / 239.0,  -239.0},
        {233, 1.0 / 233.0,  -233.0},
        {229, 1.0 / 229.0,  -229.0},
        {227, 1.0 / 227.0,  -227.0},
        {223, 1.0 / 223.0,  -223.0},
        {217, 1.0 / 217.0,  -217.0},
        {211, 1.0 / 211.0,  -211.0},
        {199, 1.0 / 199.0,  -199.0},
        {197, 1.0 / 197.0,  -197.0},
        {193, 1.0 / 193.0,  -193.0},
        {191, 1.0 / 191.0,  -191.0},
        {181, 1.0 / 181.0,  -181.0},
        {179, 1.0 / 179.0,  -179.0},
        {173, 1.0 / 173.0,  -173.0},
};

// This is deliberately private to the fused C8-only driver.  The old generic
// BlasArgs/BlasQueue definitions carried legacy C/C32 fields that this path no
// longer has or needs.
struct SmeGemmArgs {
    const void *a;
    void *b;
    const float *alpha;
    BLASLONG k;
    BLASLONG lda;
    BLASLONG ldb;
    BLASINT8 oa;
    BLASINT8 ob;
    void *common;
};

/*************** util func ********************/
// level3_thread_simple.c   // 计算任务划分，每个线程的起始点和终止点
static void GetMSimpleThreadsRegions(const BLASLONG n, BLASLONG *rangeM, const BLASLONG nthreads,
    const BLASLONG alignedSz)
{
    size_t i;
    if (nthreads == 1 || n < alignedSz) {
        i = 2; // index 2 for range of threads
        rangeM[0] = 0;
        rangeM[1] = n;
    } else {
        BLASLONG avgRange = n / nthreads;
        if (avgRange < alignedSz) {
            avgRange = alignedSz;
        }
        avgRange = ((avgRange + alignedSz - 1) / alignedSz) * alignedSz;
        BLASLONG r = 0;
        BLASLONG tmpNthreads = MIN(nthreads, n / avgRange);
        if (tmpNthreads * avgRange < n) {
            r = n - tmpNthreads * avgRange;
        }
        for (i = 0; i < tmpNthreads; ++i) {
            rangeM[i] = avgRange * i;
        }
        BLASLONG avgAddRange = ((nthreads - tmpNthreads) > 0) ? r / (nthreads - tmpNthreads) : 0;
        for (; i < nthreads; ++i) {
            rangeM[i] = rangeM[i - 1] + avgAddRange;
        }

        if (nthreads < MAX_CPU_NUMBER) {
            rangeM[nthreads] = n;
            ++i;
        }
    }
    for (; i < MAX_CPU_NUMBER; ++i) {
        rangeM[i] = rangeM[i - 1];
    }
}

// memory.c
typedef struct {
    unsigned char type;
    size_t length;
    unsigned long long cookie;
    char reserved[40];  // aligned to cacheline
} AllocHead;

#define BLAS_MEM_ALIGN 4096

/*************** util func ********************/



// level3_sme.c
#define LEVEL3_GEMM_R 8192
#define LEVEL3_GEMM_Q 2048
// P 是M维度， Q是K维度，R是N维度
// WARNING: 可以改成256
#define LEVEL3_GEMM_P 128
static void SmeGemmDriver(const SmeGemmArgs *args, FLOAT *sa, FLOAT *sb, BLASULONG mask, const BLASLONG *rangeM, const BLASLONG *rangeN, int8_t *c8, size_t ldc8, const InverseScalingConstants& scaling)
{
    
    int thread_id = omp_get_thread_num();
    int mypos = thread_id % nthreadsM;
    int nypos = thread_id / nthreadsM;
    
    BLASLONG ls, is, js;
    BLASLONG minL, minI, minJ;
    FLOAT *a = (FLOAT *)args->a;
    BLASLONG lda = args->lda;
    FLOAT *b = (FLOAT *)args->b;
    BLASLONG ldb = args->ldb;
    float *alpha = (float *)args->alpha;   // alpha = 1


    BLASLONG k = args->k;
    BLASLONG mFrom = rangeM[mypos];
    BLASLONG mTo = rangeM[mypos+1];
    BLASLONG nFrom = rangeN[nypos];
    BLASLONG nTo = rangeN[nypos+1];
    
    FLOAT *bufaa = sa;
    FLOAT *bufbb = sb;
    
    int threads = nthreadsM;
    int Jblock = 32;
    FLOAT ** bufferB = (FLOAT**)(args->common);

    for (js = nFrom; js < nTo; js += minJ) {
        //printf("NFROM %d %d\n",nFrom, nTo);
        minJ = nTo - js;
        if (minJ > LEVEL3_GEMM_R) {
            minJ = LEVEL3_GEMM_R;
        }
        for (ls = 0; ls < k; ls += minL) {
            minL = k - ls;
            if (minL >= LEVEL3_GEMM_Q) {
                minL = LEVEL3_GEMM_Q;
            }
#pragma omp barrier
            if (mask & GEMM_PB_MASK) {
                #if 0
                int minJJ = Jblock;
                int myJ = Jblock * mypos;
                PACK_B(minL, minJJ, b, ldb, ls, js + myJ, bufbb + minL * myJ, args->ob);
                int Jleft = minJ - Jblock * threads ;
                myJ = (Jleft + threads - 1) / threads;
                if (myJ * mypos +myJ > Jleft) {
                    myJ = Jleft - myJ * mypos;
                }
                PACK_B(minL, myJ, b, ldb, ls, js + Jblock * threads + myJ * mypos, bufbb + minL * (Jblock * threads + myJ * mypos) , args->ob);
                bufferB[mypos] = bufbb;   
                #else
                //printf("I pack %d, %d. from %d\n", mypos, nypos, nFrom);
                int myJ = (minJ + threads - 1) / threads;
                if (myJ * mypos > nTo) {
                    myJ = nTo - myJ * mypos;
                }
                PACK_B(minL, myJ, b, ldb, ls, js + myJ * mypos, bufbb + nypos*minL*minJ + minL * (myJ * mypos) , args->ob);
                bufferB[mypos] = bufbb; 
                #endif         
            }
#pragma omp barrier
            for (is = mFrom; is < mTo; is += minI) {
                minI = mTo - is;
                if (minI >= LEVEL3_GEMM_P) {
                    minI = LEVEL3_GEMM_P;
                } 
                
                PACK_A(minL, minI, a, lda, ls, is, bufaa, args->oa);
                
                //! start kernel
                int minJJ = Jblock;
                int start_off = js; //js + minJJ * mypos;

                

                for(int jj = start_off; jj < js + minJ && jj < nTo; jj+= minJJ) {
                    minJJ = minJ + js - jj;
                    if (minJJ > Jblock){
                        minJJ = Jblock;
                    }
                    // K=2048 is one complete K panel for this production
                    // contract.  The fused NN kernel receives this direct C8
                    // tile cursor and its unscaled byte stride, then writes C8
                    // immediately after inverse scaling each ZA vector.
                    int8_t *const c8_tile =
                        c8 + is + jj * static_cast<BLASLONG>(ldc8);
                    Int8FusedStoreParams store_params{};
                    store_params.c8 = c8_tile;
                    store_params.modulus = scaling.modulus;
                    store_params.inv_p = scaling.inv_p;
                    store_params.neg_p = scaling.neg_p;

                    KERNEL_OPERATION_SME(minI, minJJ, minL, alpha, bufaa, lda,
                                         bufbb + nypos*minL*minJ + (jj - js) * minL,
                                         ldb, c8_tile, static_cast<BLASLONG>(ldc8),
                                         &store_params);

                }
            }
        }
    }
}


void cblas_gemm_s8s8s8( const CBLAS_LAYOUT layout,
    const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb, 
    const BLASINT m, 
    const BLASINT n, 
    const BLASINT k, 
    const float alpha,
    void *a_, 
    const BLASINT lda, 
    const BLASINT8 oa,
    void *b_, 
    const BLASINT ldb, 
    const BLASINT8 ob,
    const float beta, 
    int8_t *sa,
    int8_t *sb, 
    int8_t *c8,
    size_t ldc8,
    unsigned num_moduli
)
{
    int nthreads = nthreadsM * nthreadsN;
    // 初始化一个args，包含 nthread， m
    SmeGemmArgs newArgs{};
    static const float ALPHA_ONE = 1.0f;
    newArgs.alpha = &ALPHA_ONE;

    newArgs.a = a_;
    newArgs.b = b_;
    newArgs.k = k;
    newArgs.lda = lda;
    newArgs.ldb = ldb;
    newArgs.oa = oa;
    newArgs.ob = ob;
    //计算任务划分
    BLASLONG rangeM[MAX_CPU_NUMBER + 2];
    BLASLONG rangeN[MAX_CPU_NUMBER + 2];
    GetMSimpleThreadsRegions(m, rangeM, nthreadsM, REGION_ALIGN_SIZE);
    GetMSimpleThreadsRegions(n, rangeN, nthreadsN, REGION_ALIGN_SIZE);


    BLASULONG mask = GEMM_PB_MASK | GEMM_PA_MASK;    // copy自common_level3.h
    volatile int* job_t[nthreads * nthreads];
    memset(job_t, 0, nthreads * nthreads * sizeof(int*));

    newArgs.common = (void*)job_t;
    const InverseScalingConstants& scaling = INVERSE_SCALING_CONSTANTS[num_moduli];
    /* Execute parallel computation */
    // ExecBlas(nthreads, queue);
    #pragma omp parallel for num_threads(nthreads) schedule(static) shared(sa, sb)
    for (int i = 0; i < nthreads; i++) {
        int thread_id = omp_get_thread_num();
        SmeGemmDriver(&newArgs, sa+ (LEVEL3_GEMM_Q*LEVEL3_GEMM_P * thread_id), sb, mask, rangeM, rangeN, c8, ldc8, scaling);
    }
}
