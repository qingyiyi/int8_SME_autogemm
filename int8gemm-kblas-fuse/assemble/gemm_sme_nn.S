/*
 * Production NN source: GEMM ZA accumulation followed by direct INT8 store.
 * Makefile compiles this same source twice for P2: a mode-0 low-byte epilogue
 * and a nonzero-modulus SDIV epilogue.  No duplicate .S source is created.
 */
#define INT8_INVERSE_SCALING_EPILOGUE
#include "gemm_sme_nt.S"
