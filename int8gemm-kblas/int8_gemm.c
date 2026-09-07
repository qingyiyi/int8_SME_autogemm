#include "int8_gemm.h"

#define KERNEL_OPERATION_SME(M, N, K, ALPHA, SA, LDA, SB, LDB, C, LDC, X, Y, BUF) \
    int8_sme_gemm_kernel_nn                                       \
    (M, N, K, SA, LDA, ALPHA[0], (FLOAT *)SB, LDB, (int32_t *)(C) + ((X) + (Y) * (LDC)) * COMPSIZE, LDC, BUF)

//WARNING: 可以改成16X2
int nthreadsM = 32;
int nthreadsN = 1;


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
#define LEVEL3_GEMM_P 256
static void SmeGemmDriver(const BlasArgs *args, FLOAT *sa, FLOAT *sb, BLASULONG mask, const BLASLONG *rangeM, const BLASLONG *rangeN)
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
    FLOATC *c = (FLOATC *)args->c;
    BLASLONG ldc = args->ldc;

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
                int myJ = (minJ + threads - 1) / threads;
                if (myJ * mypos > nTo) {
                    myJ = nTo - myJ * mypos;
                }
                PACK_B(minL, myJ, b, ldb, ls, js + myJ * mypos, bufbb + nypos*minL*minJ + minL * (myJ * mypos) , args->ob);
                bufferB[mypos] = bufbb;         
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
                    KERNEL_OPERATION_SME(minI, minJJ, minL, alpha, bufaa, lda, bufbb + nypos*minL*minJ + (jj - js) * minL, ldb, c, ldc, is, jj, NULL);
                }
            }
        }
    }
}


void cblas_gemm_s8s8s32( const CBLAS_LAYOUT layout, 
    const CBLAS_TRANSPOSE transa,
    const CBLAS_TRANSPOSE transb, 
    const CBLAS_OFFSET offsetc,
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
    int32_t *c, 
    const BLASINT ldc, 
    const int32_t *oc,
    int8_t *sa,
    int8_t *sb
)
{
    int nthreads = nthreadsM * nthreadsN;
    // 初始化一个args，包含 nthread， m
    BlasArgs newArgs;
    memset(&newArgs, 0, sizeof(newArgs));
    static const float ALPHA_ONE = 1.0f;
    newArgs.alpha = (void*)&ALPHA_ONE;

    newArgs.nthreads = nthreads;
    newArgs.a = a_;
    newArgs.b = b_;
    newArgs.c = c;
    newArgs.m = m;
    newArgs.n = n;
    newArgs.k = k;
    newArgs.lda = lda;
    newArgs.ldb = ldb;
    newArgs.ldc = ldc;
    newArgs.oa = oa;
    newArgs.ob = ob;
    newArgs.oc = oc;
    //计算任务划分
    BLASLONG rangeM[MAX_CPU_NUMBER + 2];
    BLASLONG rangeN[MAX_CPU_NUMBER + 2];
    GetMSimpleThreadsRegions(m, rangeM, nthreadsM, REGION_ALIGN_SIZE);
    GetMSimpleThreadsRegions(n, rangeN, nthreadsN, REGION_ALIGN_SIZE);


    BLASULONG mask = GEMM_PB_MASK | GEMM_PA_MASK;    // copy自common_level3.h
    volatile int* job_t[nthreads * nthreads];
    memset(job_t, 0, nthreads * nthreads * sizeof(int*));

    newArgs.common = (void*)job_t;
    /* Execute parallel computation */
    // ExecBlas(nthreads, queue);
    #pragma omp parallel for num_threads(nthreads) schedule(static) shared(sa, sb)
    for (int i = 0; i < nthreads; i++) {
        int thread_id = omp_get_thread_num();
        SmeGemmDriver(&newArgs, sa+ (LEVEL3_GEMM_Q*LEVEL3_GEMM_P * thread_id), sb, mask, rangeM, rangeN);
    }
}