// fused GEMM op starts here
#include <immintrin.h>

#define STORE_MATRIX 1

#define MIN(a, b) ((a) < (b)) ? (a) : (b)
#define MAX(a, b) ((a) > (b)) ? (a) : (b)

#define KERNEL_k2m4n12 \
    "vmovupd   (%0),%%ymm0; vpermilpd $5,%%ymm0,%%ymm1; prefetcht0 512(%0);"\
    "prefetcht0 128(%1); prefetcht0 128(%1,%%r12,1); prefetcht0 128(%1,%%r12,2);"\
    "vbroadcastf128   (%1),        %%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm4;  vfmadd231pd %%ymm1,%%ymm2,%%ymm5; "\
    "vbroadcastf128 16(%1),        %%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm6;  vfmadd231pd %%ymm1,%%ymm3,%%ymm7; "\
    "vbroadcastf128   (%1,%%r12,1),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm8;  vfmadd231pd %%ymm1,%%ymm2,%%ymm9; "\
    "vbroadcastf128 16(%1,%%r12,1),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm10; vfmadd231pd %%ymm1,%%ymm3,%%ymm11;"\
    "vbroadcastf128   (%1,%%r12,2),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm12; vfmadd231pd %%ymm1,%%ymm2,%%ymm13;"\
    "vbroadcastf128 16(%1,%%r12,2),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm14; vfmadd231pd %%ymm1,%%ymm3,%%ymm15;"\
    "vmovupd 32(%0),%%ymm0; vpermilpd $5,%%ymm0,%%ymm1; addq $64,%0;"\
    "vbroadcastf128 32(%1),        %%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm4;  vfmadd231pd %%ymm1,%%ymm2,%%ymm5; "\
    "vbroadcastf128 48(%1),        %%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm6;  vfmadd231pd %%ymm1,%%ymm3,%%ymm7; "\
    "vbroadcastf128 32(%1,%%r12,1),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm8;  vfmadd231pd %%ymm1,%%ymm2,%%ymm9; "\
    "vbroadcastf128 48(%1,%%r12,1),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm10; vfmadd231pd %%ymm1,%%ymm3,%%ymm11;"\
    "vbroadcastf128 32(%1,%%r12,2),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm12; vfmadd231pd %%ymm1,%%ymm2,%%ymm13;"\
    "vbroadcastf128 48(%1,%%r12,2),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm14; vfmadd231pd %%ymm1,%%ymm3,%%ymm15;"\
    "addq $64,%1;"

#define KERNEL_k1m4n12 \
    "vmovupd   (%0),%%ymm0; vpermilpd $5,%%ymm0,%%ymm1; addq $32,%0;"\
    "vbroadcastf128   (%1),        %%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm4;  vfmadd231pd %%ymm1,%%ymm2,%%ymm5; "\
    "vbroadcastf128 16(%1),        %%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm6;  vfmadd231pd %%ymm1,%%ymm3,%%ymm7; "\
    "vbroadcastf128   (%1,%%r12,1),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm8;  vfmadd231pd %%ymm1,%%ymm2,%%ymm9; "\
    "vbroadcastf128 16(%1,%%r12,1),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm10; vfmadd231pd %%ymm1,%%ymm3,%%ymm11;"\
    "vbroadcastf128   (%1,%%r12,2),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm12; vfmadd231pd %%ymm1,%%ymm2,%%ymm13;"\
    "vbroadcastf128 16(%1,%%r12,2),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm14; vfmadd231pd %%ymm1,%%ymm3,%%ymm15;"\
    "addq $32,%1;"

#if (STORE_MATRIX)

#define unit_save_m4n2(c1,c2) \
    "vunpcklpd "#c2","#c1",%%ymm2; vunpckhpd "#c1","#c2",%%ymm3;"\
    "vmovupd (%2), "#c1"; vmovupd (%2,%4,1), "#c2";"\
    "vmovupd 32(%2), %%ymm0; vmovupd 32(%2,%4,1), %%ymm1;"\
    "vfmadd231pd (%3),%%ymm2,"#c1"; vfmadd231pd (%3,%4,1),%%ymm3,"#c2";"\
    "vfmadd231pd 32(%3),%%ymm2,%%ymm0; vfmadd231pd 32(%3,%4,1),%%ymm3,%%ymm1;"\
    "vmovupd "#c1", (%2); vmovupd "#c2", (%2,%4,1);"\
    "vmovupd %%ymm0, 32(%2); vmovupd %%ymm1, 32(%2,%4,1);"
#else

#define unit_save_m4n2(c1,c2) \
    "vunpcklpd "#c2","#c1",%%ymm2; vunpckhpd "#c1","#c2",%%ymm3;"\
    "vmovupd (%2), "#c1"; vmovupd (%2,%4,1), "#c2";"\
    "vmovupd 32(%2), %%ymm0; vmovupd 32(%2,%4,1), %%ymm1;"\
    "vfmadd231pd (%3),%%ymm2,"#c1"; vfmadd231pd (%3,%4,1),%%ymm3,"#c2";"\
    "vfmadd231pd 32(%3),%%ymm2,%%ymm0; vfmadd231pd 32(%3,%4,1),%%ymm3,%%ymm1;"
#endif

// %ymm4 to hold real, %ymm5 to hold imag

#define SAVE_m4n12 \
    unit_save_m4n2(%%ymm4,%%ymm5)\
    "vaddpd %%ymm4, %%ymm5, %%ymm4; vaddpd %%ymm0, %%ymm1, %%ymm5;"\
    "leaq (%2,%4,2),%2;leaq (%3,%4,2),%3;"\
    unit_save_m4n2(%%ymm6,%%ymm7)\
    "vaddpd %%ymm0, %%ymm1, %%ymm1; vaddpd %%ymm6, %%ymm7, %%ymm0;"\
    "vaddpd %%ymm0, %%ymm4, %%ymm4; vaddpd %%ymm1, %%ymm5, %%ymm5;"\
    "leaq (%2,%4,2),%2;leaq (%3,%4,2),%3;"\
    unit_save_m4n2(%%ymm8,%%ymm9)\
    "vaddpd %%ymm0, %%ymm1, %%ymm1; vaddpd %%ymm8, %%ymm9, %%ymm0;"\
    "vaddpd %%ymm0, %%ymm4, %%ymm4; vaddpd %%ymm1, %%ymm5, %%ymm5;"\
    "leaq (%2,%4,2),%2;leaq (%3,%4,2),%3;"\
    unit_save_m4n2(%%ymm10,%%ymm11)\
    "vaddpd %%ymm0, %%ymm1, %%ymm1; vaddpd %%ymm10, %%ymm11, %%ymm0;"\
    "vaddpd %%ymm0, %%ymm4, %%ymm4; vaddpd %%ymm1, %%ymm5, %%ymm5;"\
    "leaq (%2,%4,2),%2;leaq (%3,%4,2),%3;"\
    unit_save_m4n2(%%ymm12,%%ymm13)\
    "vaddpd %%ymm0, %%ymm1, %%ymm1; vaddpd %%ymm12, %%ymm13, %%ymm0;"\
    "vaddpd %%ymm0, %%ymm4, %%ymm4; vaddpd %%ymm1, %%ymm5, %%ymm5;"\
    "leaq (%2,%4,2),%2;leaq (%3,%4,2),%3;"\
    unit_save_m4n2(%%ymm14,%%ymm15)\
    "vaddpd %%ymm0, %%ymm1, %%ymm1; vaddpd %%ymm14, %%ymm15, %%ymm0;"\
    "vaddpd %%ymm0, %%ymm4, %%ymm4; vaddpd %%ymm1, %%ymm5, %%ymm5;"\
    "salq $1,%4;subq %4,%2;subq %4,%3;salq $2,%4;subq %4,%2;subq %4,%3;sarq $3,%4;addq $64,%2;addq $64,%3;"\
    "vmovsd (%10), %%xmm0; vmovsd 8(%10), %%xmm1;"\
    "vextractf128 $1, %%ymm4, %%xmm12;vextractf128 $1, %%ymm5, %%xmm13;"\
    "vaddpd %%xmm4, %%xmm12, %%xmm4; vaddpd %%xmm5, %%xmm13, %%xmm5;"\
    "vhaddpd %%xmm4, %%xmm4, %%xmm4; vhaddpd %%xmm5, %%xmm5, %%xmm5;"\
    "vaddpd %%xmm0, %%xmm4, %%xmm4; vaddpd %%xmm1, %%xmm5, %%xmm5;"\
    "vmovsd %%xmm4, (%10); vmovsd %%xmm5, 8(%10);"

#define KERNEL_k1m4n8 \
    "vmovupd   (%0),%%ymm0; vpermilpd $5,%%ymm0,%%ymm1; addq $32,%0;"\
    "vbroadcastf128   (%1),        %%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm4;  vfmadd231pd %%ymm1,%%ymm2,%%ymm5; "\
    "vbroadcastf128 16(%1),        %%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm6;  vfmadd231pd %%ymm1,%%ymm3,%%ymm7; "\
    "vbroadcastf128   (%1,%%r12,1),%%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm8;  vfmadd231pd %%ymm1,%%ymm2,%%ymm9; "\
    "vbroadcastf128 16(%1,%%r12,1),%%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm10; vfmadd231pd %%ymm1,%%ymm3,%%ymm11;"\
    "addq $32,%1;"

#define KERNEL_k2m4n8 KERNEL_k1m4n8 KERNEL_k1m4n8

#define SAVE_m4n8 \
    unit_save_m4n2(%%ymm4,%%ymm5)\
    "vaddpd %%ymm4, %%ymm5, %%ymm4; vaddpd %%ymm0, %%ymm1, %%ymm5;"\
    "leaq (%2,%4,2),%2;leaq (%3,%4,2),%3;"\
    unit_save_m4n2(%%ymm6,%%ymm7)\
    "vaddpd %%ymm0, %%ymm1, %%ymm1; vaddpd %%ymm6, %%ymm7, %%ymm0;"\
    "vaddpd %%ymm0, %%ymm4, %%ymm4; vaddpd %%ymm1, %%ymm5, %%ymm5;"\
    "leaq (%2,%4,2),%2;leaq (%3,%4,2),%3;"\
    unit_save_m4n2(%%ymm8,%%ymm9)\
    "vaddpd %%ymm0, %%ymm1, %%ymm1; vaddpd %%ymm8, %%ymm9, %%ymm0;"\
    "vaddpd %%ymm0, %%ymm4, %%ymm4; vaddpd %%ymm1, %%ymm5, %%ymm5;"\
    "leaq (%2,%4,2),%2;leaq (%3,%4,2),%3;"\
    unit_save_m4n2(%%ymm10,%%ymm11)\
    "vaddpd %%ymm0, %%ymm1, %%ymm1; vaddpd %%ymm10, %%ymm11, %%ymm0;"\
    "vaddpd %%ymm0, %%ymm4, %%ymm4; vaddpd %%ymm1, %%ymm5, %%ymm5;"\
    "salq $1,%4;subq %4,%2;subq %4,%3;salq $1,%4;subq %4,%2;subq %4,%3;sarq $2,%4;addq $64,%2;addq $64,%3;"\
    "vmovsd (%10), %%xmm0; vmovsd 8(%10), %%xmm1;"\
    "vextractf128 $1, %%ymm4, %%xmm12;vextractf128 $1, %%ymm5, %%xmm13;"\
    "vaddpd %%xmm4, %%xmm12, %%xmm4; vaddpd %%xmm5, %%xmm13, %%xmm5;"\
    "vhaddpd %%xmm4, %%xmm4, %%xmm4; vhaddpd %%xmm5, %%xmm5, %%xmm5;"\
    "vaddpd %%xmm0, %%xmm4, %%xmm4; vaddpd %%xmm1, %%xmm5, %%xmm5;"\
    "vmovsd %%xmm4, (%10); vmovsd %%xmm5, 8(%10);"

#define KERNEL_k1m4n4 \
    "vmovupd   (%0),%%ymm0; vpermilpd $5,%%ymm0,%%ymm1; addq $32,%0;"\
    "vbroadcastf128   (%1),        %%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm4;  vfmadd231pd %%ymm1,%%ymm2,%%ymm5; "\
    "vbroadcastf128 16(%1),        %%ymm3; vfmadd231pd %%ymm0,%%ymm3,%%ymm6;  vfmadd231pd %%ymm1,%%ymm3,%%ymm7; "\
    "addq $32,%1;"

#define KERNEL_k2m4n4 KERNEL_k1m4n4 KERNEL_k1m4n4

#define SAVE_m4n4 \
    unit_save_m4n2(%%ymm4,%%ymm5)\
    "vaddpd %%ymm4, %%ymm5, %%ymm4; vaddpd %%ymm0, %%ymm1, %%ymm5;"\
    "leaq (%2,%4,2),%2;leaq (%3,%4,2),%3;"\
    unit_save_m4n2(%%ymm6,%%ymm7)\
    "vaddpd %%ymm0, %%ymm1, %%ymm1; vaddpd %%ymm6, %%ymm7, %%ymm0;"\
    "vaddpd %%ymm0, %%ymm4, %%ymm4; vaddpd %%ymm1, %%ymm5, %%ymm5;"\
    "subq %4,%2;subq %4,%2;subq %4,%3;subq %4,%3;addq $64,%2;addq $64,%3;"\
    "vmovsd (%10), %%xmm0; vmovsd 8(%10), %%xmm1;"\
    "vextractf128 $1, %%ymm4, %%xmm12;vextractf128 $1, %%ymm5, %%xmm13;"\
    "vaddpd %%xmm4, %%xmm12, %%xmm4; vaddpd %%xmm5, %%xmm13, %%xmm5;"\
    "vhaddpd %%xmm4, %%xmm4, %%xmm4; vhaddpd %%xmm5, %%xmm5, %%xmm5;"\
    "vaddpd %%xmm0, %%xmm4, %%xmm4; vaddpd %%xmm1, %%xmm5, %%xmm5;"\
    "vmovsd %%xmm4, (%10); vmovsd %%xmm5, 8(%10);"

#define KERNEL_k1m4n2 \
    "vmovupd   (%0),%%ymm0; vpermilpd $5,%%ymm0,%%ymm1; addq $32,%0;"\
    "vbroadcastf128   (%1),        %%ymm2; vfmadd231pd %%ymm0,%%ymm2,%%ymm4;  vfmadd231pd %%ymm1,%%ymm2,%%ymm5; "\
    "addq $16,%1;"

#define KERNEL_k2m4n2 KERNEL_k1m4n2 KERNEL_k1m4n2

#define SAVE_m4n2 \
    unit_save_m4n2(%%ymm4,%%ymm5)\
    "vaddpd %%ymm4, %%ymm5, %%ymm4; vaddpd %%ymm0, %%ymm1, %%ymm5;"\
    "addq $64,%2;addq $64,%3;"\
    "vmovsd (%10), %%xmm0; vmovsd 8(%10), %%xmm1;"\
    "vextractf128 $1, %%ymm4, %%xmm12;vextractf128 $1, %%ymm5, %%xmm13;"\
    "vaddpd %%xmm4, %%xmm12, %%xmm4; vaddpd %%xmm5, %%xmm13, %%xmm5;"\
    "vhaddpd %%xmm4, %%xmm4, %%xmm4; vhaddpd %%xmm5, %%xmm5, %%xmm5;"\
    "vaddpd %%xmm0, %%xmm4, %%xmm4; vaddpd %%xmm1, %%xmm5, %%xmm5;"\
    "vmovsd %%xmm4, (%10); vmovsd %%xmm5, 8(%10);"

#define INIT_m4n2 "vpxor %%ymm4,%%ymm4,%%ymm4; vpxor %%ymm5,%%ymm5,%%ymm5;"
#define INIT_m4n4 INIT_m4n2 "vpxor %%ymm6,%%ymm6,%%ymm6; vpxor %%ymm7,%%ymm7,%%ymm7;"
#define INIT_m4n8 INIT_m4n4 "vpxor %%ymm8,%%ymm8,%%ymm8; vpxor %%ymm9,%%ymm9,%%ymm9; vpxor %%ymm10,%%ymm10,%%ymm10; vpxor %%ymm11,%%ymm11,%%ymm11;"
#define INIT_m4n12 INIT_m4n8 "vpxor %%ymm12,%%ymm12,%%ymm12; vpxor %%ymm13,%%ymm13,%%ymm13; vpxor %%ymm14,%%ymm14,%%ymm14; vpxor %%ymm15,%%ymm15,%%ymm15;"

#define KERNEL_m4(ndim) \
    "movq %2,%5;movq %3,%7;cmpq $24,%8;jb 74"#ndim"1f;"\
    "74"#ndim"0:\n\t"\
    "prefetcht1 (%7); prefetcht1 31(%7); addq %4,%7;"\
    KERNEL_k2m4n##ndim\
    KERNEL_k2m4n##ndim\
    KERNEL_k2m4n##ndim\
    "prefetcht1 (%5); prefetcht1 31(%5); addq %4,%5;"\
    KERNEL_k2m4n##ndim\
    KERNEL_k2m4n##ndim\
    KERNEL_k2m4n##ndim\
    "prefetcht1 (%6); addq $10,%6;"\
    "subq $12,%8;cmpq $24,%8;jnb 74"#ndim"0b;"\
    "movq %2,%5;movq %3,%7;"\
    "74"#ndim"1:\n\t"\
    "cmpq $1,%8;jb 74"#ndim"2f;"\
    "prefetcht0 (%5); prefetcht0 31(%5); addq %4,%5;"\
    "prefetcht0 (%7); prefetcht0 31(%7); addq %4,%7;"\
    KERNEL_k1m4n##ndim\
    "decq %8;jmp 74"#ndim"1b;"\
    "74"#ndim"2:\n\t"\
    "movq %%r13,%8; movq %%r11,%1;"\
    "prefetcht0 (%1);prefetcht0 64(%1);"

//%0 -> a; %1 -> b; %2 -> c; %3 -> c_fused; %4 = ldc(bytes); %8 = k_count, %9 = m_count, %5 = c_pref, %6 = b_pref;
//cases with n=8,12 requires r12 for efficient addressing. r12 = k << 5; r13 for k, r14 for m, r15 for a_head_address; r11 for b_head_address;
#define COMPUTE(ndim) {\
    b_pref = b_pointer + ndim * K;\
    __asm__ __volatile__(\
    "movq %1,%%r11; movq %8,%%r13; movq %9,%%r14; movq %0,%%r15; movq %8,%%r12; salq $5,%%r12; movq %7,%%r10;"\
    "cmpq $4,%9;jb "#ndim"01f;"\
    #ndim"00:\n\t"\
    INIT_m4n##ndim\
    KERNEL_m4(ndim)\
    SAVE_m4n##ndim\
    "subq $4,%9;cmpq $4,%9;jnb "#ndim"00b;"\
    #ndim"01:\n\t"\
    "movq %%r14,%9;salq $3,%%r14;subq %%r14,%2;subq %%r14,%2;subq %%r14,%3;subq %%r14,%3;movq %%r15,%0;vzeroupper;"\
    :"+r"(a_pointer),"+r"(b_pointer),"+r"(c_pointer),"+r"(c_fuse_pointer),"+r"(ldc_in_bytes),"+r"(c_pref),"+r"(b_pref), "+r"(c_fuse_pref):"m"(K),"m"(M), "r"(sum_ptr)\
    :"xmm0","xmm1","xmm2","xmm3","xmm4","xmm5","xmm6","xmm7","xmm8","xmm9","xmm10","xmm11","xmm12","xmm13","xmm14","xmm15","r11","r12","r13","r14","r15","cc","memory");\
    b_pointer += K * ndim; c_pointer += ldc * ndim; c_fuse_pointer += ldc * ndim;\
}


//#include "common.h"
//#include <stdint.h>
int __attribute__ ((noinline)) KERNEL(int m, int n, int k, double * __restrict__ A, double * __restrict__ B, double * __restrict__ C, int ldc, double * __restrict__ fuse_C, double * __restrict__ sum){
    int64_t ldc_in_bytes = (int64_t)ldc * sizeof(double), M = (int64_t)m, K = (int64_t)k;
    double *a_pointer = A, *b_pointer = B, *c_pointer = C, *c_pref = C,*b_pref = B, *c_fuse_pointer = fuse_C, *c_fuse_pref = fuse_C, *sum_ptr = sum;
    int ndim_count = n;
    for(;ndim_count>11;ndim_count-=12) COMPUTE(12)
    for(;ndim_count>7;ndim_count-=8) COMPUTE(8)
    for(;ndim_count>3;ndim_count-=4) COMPUTE(4)
    for(;ndim_count>1;ndim_count-=2) COMPUTE(2)
    return 0;
}

/* test zone */
static void dgemm_tcopy_4(double alpha, double *src, double *dst, int lead_dim, int dim_first, int dim_second){
//src_leading_dim parallel with dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    int count_first,count_second;
    double *tosrc,*todst;
    __m256d valpha_256 = _mm256_set1_pd(alpha);
    __m128d valpha_128 = _mm_set1_pd(alpha);
    for(count_second=0;count_second<dim_second;count_second++){
      tosrc = src + count_second * lead_dim;
      todst = dst + count_second * 4;
      for(count_first=dim_first;count_first>3;count_first-=4){
        _mm256_storeu_pd(todst, _mm256_mul_pd(_mm256_loadu_pd(tosrc), valpha_256));
        tosrc+=4;todst+=4*dim_second;
      }
      todst -= count_second * 2;
      for(;count_first>1;count_first-=2){
        _mm_storeu_pd(todst,_mm_mul_pd(_mm_loadu_pd(tosrc), valpha_128));
        tosrc+=2;todst+=2*dim_second;
      }
      todst -= count_second;
      if(count_first>0) *todst=(*tosrc)*alpha;
    }
}
static void dgemm_ncopy_4(double *src, double *dst, int lead_dim, int dim_first, int dim_second){
//src_leading_dim perpendicular to dst_tile_leading_dim
    if(dim_first==0 || dim_second==0) return;
    int count_first,count_second,tosrc_inc;
    double *tosrc1,*tosrc2,*tosrc3,*tosrc4;
    double *todst=dst;
    tosrc1=src;tosrc2=tosrc1+lead_dim;tosrc3=tosrc2+lead_dim;tosrc4=tosrc3+lead_dim;
    tosrc_inc=4*lead_dim-dim_first;
    for(count_second=dim_second;count_second>3;count_second-=4){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst[2]=*tosrc3;tosrc3++;todst[3]=*tosrc4;tosrc4++;
        todst+=4;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;tosrc3+=tosrc_inc;tosrc4+=tosrc_inc;
    }
    tosrc_inc-=2*lead_dim;
    for(;count_second>1;count_second-=2){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;todst[1]=*tosrc2;tosrc2++;
        todst+=2;
      }
      tosrc1+=tosrc_inc;tosrc2+=tosrc_inc;
    }
    if(count_second>0){
      for(count_first=0;count_first<dim_first;count_first++){
        todst[0]=*tosrc1;tosrc1++;
        todst++;
      }
    }
}
static void SCALE_MULT(double *dat,double *sca, int lead_dim, int dim_first, int dim_second){
//dim_first parallel with leading dim; dim_second perpendicular to leading dim.
    if(dim_first==0 || dim_second==0 || (*sca)==1.0) return;
    double scale = *sca; double *current_dat = dat;
    int count_first,count_second;
    for(count_second=0;count_second<dim_second;count_second++){
      for(count_first=0;count_first<dim_first;count_first++){
        *current_dat *= scale; current_dat++;
      }
      current_dat += lead_dim - dim_first;
    }
}
#define BLOCKDIM_K 240 //GEMM_Q in OpenBLAS
#define BLOCKDIM_M 512 //GEMM_P in OpenBLAS
#define BLOCKDIM_N 9216
#define NOTRANSA ((*transa)=='N'||(*transa)=='n')
#define NOTRANSB ((*transb)=='N'||(*transb)=='n')
//gcc -march=haswell --shared -fPIC -O2 dgemm_kernel_4x4_haswell.c -o dgemm.so

static inline void partition_m_dim(const int ithr, const int nthrs, const int m, int *offset, int *block)
{
    // int band; 
    // int tail; 

    int band = m / nthrs;
    int tail = m - (nthrs - 1) * band;
    if (tail > (band + 1)){
        band++;
    }

    if (band % 4 != 0) {
        band = (m / nthrs) & -4;
    }
    tail = m - (nthrs - 1) * band;

    if (ithr < (nthrs - 1)){
        *block = band;
    }else{
        *block = tail;
    }
    *offset = ithr * band;
    if (*offset >= m) {
        *block = 0;
        *offset = 0;
    }else if ((*offset + *block) > m) {
        *block = m - *offset;
    }
}

static inline int div_up(int a, int b){
    return ((a + b - 1) / b);
}

static inline int rnd_up(int a, int b) {
    return (div_up(a, b) * b);
}


static inline int get_n_padd(int n, int un, int bn)
{
    return rnd_up(MIN(MAX(n, un), bn), un);
}

static inline int get_n_padd_parallel_a(int n, int nthr)
{
    int n_padd = get_n_padd(n, 8, BLOCKDIM_N);
    return n_padd;
}

#define A(i,j) A[(i)+(j)*LDA]
#define B(i,j) B[(i)+(j)*LDB]
#define C(i,j) C[(i)+(j)*LDC]
#define fuse_C(i, j) fuse_C[(i)+(j)*LDC]

void fused_dgemm(int *m,int *n,int *k,double *alpha,double *a,int *lda,double *b,int *ldb,double *c,int *ldc,double *fuse_C, double *sum){
    int M = *m, N = *n, K = *k;
    int LDA = *lda, LDB = *ldb, LDC = *ldc;
    double *A = a, *B = b, *C = c;
    double *b_buffer_global = NULL;

    #pragma omp parallel
    {
        int nthr = omp_get_num_threads();
        int ithr = omp_get_thread_num();
        
        int m_offset = 0, m_block = 0;
        partition_m_dim(ithr, nthr, M, &m_offset, &m_block);

        int m_count, n_count, k_count;
        int m_inc, n_inc, k_inc;
        double *a_buffer_local = NULL;
        double *b_buffer_local = NULL;
        if (ithr == 0) {
            b_buffer_global = (double *)aligned_alloc(4096, sizeof(double) * (BLOCKDIM_N * BLOCKDIM_K));
        }
        #pragma omp barrier
        b_buffer_local = b_buffer_global;
        for (k_count = 0; k_count < K; k_count += k_inc){
            k_inc = (K - k_count > BLOCKDIM_K) ? BLOCKDIM_K : K - k_count;
            // parallel copy for A packing
            for (n_count = 0; n_count < N; n_count += n_inc) {
                n_inc = (N - n_count > BLOCKDIM_N) ? BLOCKDIM_N : N - n_count;
                int band = (n_inc + nthr - 1) / nthr;
                band = rnd_up(band, 12);
                int offset = band * ithr;
                if (offset > n_inc) {
                    offset = 0;
                    band = 0;
                }
                if (offset + band > n_inc) {
                    band = n_inc - offset;
                }
                if (band > 0) {
                    dgemm_ncopy_4(&B(k_count, n_count + offset), b_buffer_local + offset * k_inc, LDB, k_inc, band);
                }
                #pragma omp barrier
                if (!a_buffer_local) {
                    a_buffer_local = (double *)aligned_alloc(4096, sizeof(double) * (BLOCKDIM_M * k_inc));
                }

                for (m_count = 0; m_count < m_block; m_count += m_inc) {
                    m_inc = (m_block - m_count > BLOCKDIM_M) ? BLOCKDIM_M : m_block - m_count;
                    dgemm_tcopy_4(*alpha, &A(m_offset + m_count, k_count), a_buffer_local, LDA, m_inc, k_inc);
                    double *b_buff_ptr = b_buffer_local;
                    // macro_kernel_k19(a_buffer_local, b_buff_ptr, m_inc, n_inc, k_inc, &C(m_count + m_offset, n_count), LDC);
                    KERNEL(m_inc, n_inc, k_inc, a_buffer_local, b_buff_ptr, &C((m_count + m_offset) * 2, n_count), LDC, &fuse_C((m_count + m_offset) * 2, n_count), sum + ithr * 2);
                }

                #pragma omp barrier
            }
        }

        if (a_buffer_local) {
            free(a_buffer_local);
        }
    }

    if (b_buffer_global) {
        free(b_buffer_global);
    }

}