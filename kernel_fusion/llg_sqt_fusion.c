#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include "gemm_kernel.h"
#include "v3d.h"

int N = 100; 
int num_neighbor = 4;
double J1 = 1.0; 
int MCsweep = 500; 
double dt = 0.02; // step size in time
int Nfile = 50; 
int trd = 1;
int vx = 5, vy = 5;
int nn[4][2]; 

inline static int min(int _x, int _y)
{
    if(_x < _y) return _x; 
    else return _y; 
}


void euler(double *spin0, double *derivative)
{
    
    memset(derivative, 0, Nfile*N*N*3*sizeof(double)); 

    #pragma omp parallel
    {

        int row, col, nr, nc;
        int cur_idx, nn_idx;  

        v3d curspin, exg_field, tmp; 
        double tx, ty, tz; 

        
        for(int id = 0; id < Nfile; id++)
        {
            
            #pragma omp for
            for(int i=0;i<N*N;i++)
            {
                row = i/N, col = i%N; 

                exg_field = make_v3d(0.0, 0.0, 0.0);

                cur_idx = id*N*N*3 + i*3; 

                curspin = make_v3d(spin0[cur_idx], spin0[cur_idx + 1], spin0[cur_idx + 2]);


                for(int j=0;j<num_neighbor;j++)
                {

                    nr = (row + nn[j][0] + N)%N; 
                    nc = (col + nn[j][1] + N)%N; 
                    // pos = nr * N + nc; 

                    nn_idx = id*N*N*3 + (nr*N + nc)*3;

                    tx = spin0[nn_idx]; 
                    ty = spin0[nn_idx + 1]; 
                    tz = spin0[nn_idx + 2]; 
                    // nnspin = make_v3d(spin0[id*N*N*3 + pos*3], spin0[id*N*N*3 + pos*3 + 1], spin0[id*N*N*3 + pos*3 + 2]); 
                    
                    exg_field.x += (tx * J1); 
                    exg_field.y += (ty * J1); 
                    exg_field.z += (tz * J1); 

                }

                // exg_field.z += (-1.0 * hz) ;

                v3d_cross(&tmp, &exg_field, &curspin); 
                

                derivative[cur_idx] = tmp.x ;
                derivative[cur_idx+1] = tmp.y ;
                derivative[cur_idx+2] = tmp.z ;
                
            }
        }        
    } 
    
}




void runge_kutta(double *spin, double *cur_input, double *k1, double *k2, double *k3, double *k4)
{
    
    #pragma omp parallel for
    for(int j=0;j<Nfile*N*N*3;j++)
        cur_input[j] = spin[j];
    
    euler(cur_input, k1);
    
    #pragma omp parallel for
    for(int j=0;j<Nfile*N*N*3;j++)
    {
        k1[j] *= dt;
        cur_input[j] = spin[j] + k1[j]/2.0;
    }
        
    
    euler(cur_input, k2); 
    
    #pragma omp parallel for
    for(int j=0;j<Nfile*N*N*3;j++)
    {
        k2[j] *= dt;
        cur_input[j] = spin[j] + k2[j]/2.0;   
    }
        
    
    euler(cur_input, k3); 
    
    #pragma omp parallel for
    for(int j=0;j<Nfile*N*N*3;j++)
    {
        k3[j] *= dt;
        cur_input[j] = spin[j] + k3[j];
        
    }
    
    euler(cur_input, k4); 
    
    #pragma omp parallel for
    for(int j=0;j<Nfile*N*N*3;j++)
    {
        k4[j] *= dt;
        cur_input[j] = spin[j] + k1[j]/6.0 + k2[j]/3.0 + k3[j]/3.0+ k4[j]/6.0;
    }
        

}



int main(int argc, char *argv[])
{
    
    
    
    char ch; 
    while((ch = getopt(argc, argv, "n:m:f:e:t:x:y:")) != EOF)
    {
        switch(ch)
        {
            case 'n' :N = atoi(optarg);
            break; 
            case 'm' : MCsweep = atoi(optarg); 
            break;
            case 'e' : dt = atof(optarg); 
            break;
            case 't' : trd = atoi(optarg); 
            break;
            case 'f' : Nfile = atoi(optarg); 
            break;
            case 'x' : vx = atoi(optarg); 
            break;
            case 'y' : vy = atoi(optarg); 
            break;

        }
    }
    
    
    vx = N / 10; 
    vy = N / 10;
    
    nn[0][0] = 1, nn[0][1] = 0; 
    nn[1][0] = -1, nn[1][1] = 0; 
    nn[2][0] = 0, nn[2][1] = -1; 
    nn[3][0] = 0, nn[3][1] = 1; 

    FILE *fp = NULL;

    char  s_tmp[20], file_name[128]; 
    
    
    
    double* spin_all; 
    spin_all = (double*)malloc(Nfile*N*N*3*sizeof(double)); 
    
    srand(time(NULL)); 

    for(int id = 0; id < Nfile;id++)
    {
        double tx, ty, tz, nrm; 
        int idx; 
        
        for(int i=0;i<N*N;i++)
        {
            tx = rand()/RAND_MAX; 
            ty = rand()/RAND_MAX; 
            tz = rand()/RAND_MAX;

            nrm = sqrt(tx*tx + ty * ty + tz * tz);

            idx = id*N*N*3 + i*3; 

            spin_all[idx] = tx / nrm; 
            spin_all[idx + 1] = ty / nrm; 
            spin_all[idx + 2] = tz / nrm;   
        }
    }

    printf("Spin configurations generated! \n");
     printf("N: %d, Nfile: %d\n", N, Nfile);

    int  tot = omp_get_max_threads(); 
    
    omp_set_num_threads( min(128, tot) ); 
    printf("Threads: %d\n", min(128, tot));
    
   
    double *zerot_avg, *fint_avg; 
    double *spin_zero; 
    
    zerot_avg = (double*)malloc(N*N*3*sizeof(double)); 
    fint_avg = (double*)malloc(N*N*3*sizeof(double)); 
    spin_zero = (double*)malloc(Nfile*N*N*3*sizeof(double)); 
    

    #pragma omp parallel for
    for(int i=0;i<N*N;i++)
    {
        for(int j=0;j<Nfile;j++)
        {
            spin_zero[i*Nfile*3 + j*3] = spin_all[j*N*N*3 + i*3]; 
            spin_zero[i*Nfile*3 + j*3 + 1] = spin_all[j*N*N*3 + i*3 +1]; 
            spin_zero[i*Nfile*3 + j*3 + 2] = spin_all[j*N*N*3 + i*3 +2]; 
                
        }
    }    
    
 
    
    #pragma omp parallel for
    for(int i=0;i<N*N;i++)
    {
        double tx=0.0, ty=0.0, tz = 0.0;
        
        for(int j=0;j<Nfile;j++)
        {
            tx += spin_zero[i*Nfile*3 + j*3]; 
            ty += spin_zero[i*Nfile*3 + j*3 + 1];
            tz += spin_zero[i*Nfile*3 + j*3 + 2]; 
        }
        
        tx /= (double)Nfile;
        ty /= (double)Nfile; 
        tz /= (double)Nfile; 

        zerot_avg[i*3] = tx;
        zerot_avg[i*3+1] = ty; 
        zerot_avg[i*3+2] = tz;
    }
    
    
    double complex *sqt; 
    sqt = (double complex*)malloc( MCsweep * sizeof(double complex));
    
    
    double *spint_mkl; 
    spint_mkl = (double*)malloc(N*N*Nfile*3*sizeof(double)); 
    
    
    double *k1, *k2, *k3, *k4, *cur_input; 
    
    k1 = (double*)malloc(N*N*Nfile*3*sizeof(double)); 
    k2 = (double*)malloc(N*N*Nfile*3*sizeof(double)); 
    k3 = (double*)malloc(N*N*Nfile*3*sizeof(double)); 
    k4 = (double*)malloc(N*N*Nfile*3*sizeof(double)); 
    cur_input = (double*)malloc(N*N*Nfile*3*sizeof(double)); 
    
    double sum[128*2];
    memset(sum, 0, sizeof(sum));
    
    double qx, qy;
    
    qx = 2.0 * M_PI*vx/(double)N; 
    qy = 2.0 * M_PI*vy/(double)N; 
    
    double alpha = 1.0/(double)Nfile, beta = 0.0;
    
    
    int mm = N*N, nn = N*N, kk = Nfile * 3; 
    int lda = mm, ldb = kk, ldc = mm*2; 
    
    double *cmat_mkl;
    cmat_mkl = (double*)malloc( 2*mm*nn * sizeof(double));
    
    double *qmat_mkl; 
    qmat_mkl = (double*)malloc( 2*mm*nn * sizeof(double));
    
    
    #pragma omp parallel for
    for(int j=0;j<N*N;j++)
    {
        double complex tmp;
        int ri, rj, ci, cj; 
        int mapped_row_real, mapped_row_imag, mapped_idx_real, mapped_idx_imag; 
        double dffr, dffc; 
        
        for(int i=0;i<N*N;i++)
        {
            ri = i/N, ci = i%N; 
            rj = j/N, cj = j%N; 
            dffr = ri - rj; 
            dffc = ci - cj;
            tmp = cexp( CMPLX(0.0, -1.0) * (qx * dffr + qy * dffc ) ); 
            
            mapped_row_real = (i / 4) * 8 + (i % 4);
            mapped_row_imag = (i / 4) * 8 + (i % 4) + 4;
            mapped_idx_real = mapped_row_real + j * 2 * mm;
            mapped_idx_imag = mapped_row_imag + j * 2 * mm;
            
            qmat_mkl[mapped_idx_real] = creal(tmp); 
            qmat_mkl[mapped_idx_imag] = cimag(tmp); 
//             qmat_mkl[i+j*N*N] = cexp( CMPLX(0.0, -1.0) * (qx * dffr + qy * dffc ) );
        }
    }
    
    double tlle = 0.0, tsqt = 0.0; 
    double t1, t2, t3;
    
    
    
    for(int num=0;num<60;num++)
    {
        
        if(num%5 == 0) printf("Step: %d\n", num);
        
        t1 = omp_get_wtime();
   
       
        runge_kutta(spin_all, cur_input, k1, k2, k3, k4);

        #pragma omp parallel for
        for(int j=0;j<Nfile*N*N;j++)
        {
            double tx = cur_input[j*3], ty = cur_input[j*3+1], tz = cur_input[j*3+2]; 
            double nrm = sqrt(tx*tx + ty * ty + tz * tz); 

            tx = 1.5 * tx / nrm; 
            ty = 1.5 * ty /nrm; 
            tz = 1.5 * tz /nrm; 

            spin_all[j*3] = tx,  spin_all[j*3+1] = ty, spin_all[j*3+2] = tz; 
        }
        
        
        t2 = omp_get_wtime();
        if(num >= 20) tlle += (t2 - t1);
        // printf("Elapsed time  of LLG:  %lf\n", t2-t1); 
        
        t1 = omp_get_wtime();
        
        #pragma omp parallel for
        for(int j=0;j<N*N;j++)
            for(int i=0;i<Nfile;i++)
            {
                spint_mkl[j + (i*3)*N*N] = spin_all[i*N*N*3 + j*3];
                spint_mkl[j + (i*3+1)*N*N] = spin_all[i*N*N*3 +j*3+1];
                spint_mkl[j + (i*3+2)*N*N] = spin_all[i*N*N*3 +j*3+2];
            }
        
        
        #pragma omp parallel for
        for(int i=0;i<N*N;i++)
        {
            double tx=0.0, ty=0.0, tz = 0.0;
            double tmp_sum; 

            for(int id = 0;id<Nfile;id++)
            {
                tx += spin_all[id * N*N*3 + i*3]; 
                ty += spin_all[id * N*N*3 + i*3+1]; 
                tz += spin_all[id * N*N*3 + i*3+2]; 
            }

            tx /= (double)Nfile; 
            ty /= (double)Nfile; 
            tz /= (double)Nfile; 
            
            fint_avg[i*3] = tx, fint_avg[i*3+1] = ty, fint_avg[i*3+2] = tz; 
        }

        
        
        #pragma omp parallel for
        for(int j=0;j<N*N;j++)
        {
            double zx = zerot_avg[j*3], zy = zerot_avg[j*3+1], zz = zerot_avg[j*3+2];
            double tx, ty, tz;
            double tmp_sum; 
            int mapped_row_real, mapped_row_imag, mapped_idx_real, mapped_idx_imag; 
            
            for(int i=0;i<N*N;i++)
            {
                tx = fint_avg[i*3], ty = fint_avg[i*3+1], tz = fint_avg[i*3+2]; 
                tmp_sum = zx * tx + zy * ty + zz * tz; 
                tmp_sum *= -1.0; 
                
                mapped_row_real = (i / 4) * 8 + (i % 4);
                mapped_row_imag = (i / 4) * 8 + (i % 4) + 4;
                mapped_idx_real = mapped_row_real + j * 2 * mm;
                mapped_idx_imag = mapped_row_imag + j * 2 * mm;

                cmat_mkl[mapped_idx_real] = tmp_sum * qmat_mkl[mapped_idx_real]; 
                cmat_mkl[mapped_idx_imag] = tmp_sum * qmat_mkl[mapped_idx_imag]; 
                
            }  
        
        }
         
        
        fused_dgemm(&mm,&nn,&kk,&alpha, spint_mkl, &lda, spin_zero, &ldb, cmat_mkl, &ldc,qmat_mkl,sum);
        
        
        double complex res = CMPLX(0.0, 0.0); 
        
        for (int i = 0; i < 128; i++) 
        {
            res += CMPLX(sum[2*i], sum[2*i+1]);
            sum[2*i] = 0.0, sum[2*i+1] = 0.0; 
        }
        

        res /= (double)(N*N); 
        sqt[num] = res; 
        
        t2 = omp_get_wtime(); 
        
        if(num >= 20) tsqt += (t2 - t1);
        
    
    }
    
    tlle /= 40.0; 
    tsqt /= 40.0;
    
    
    fp = fopen("amd_gcc_11.2_fusion_sqt.txt", "a");
    fprintf(fp, "%d %lf %lf\n", N, tlle, tsqt);
    
    fclose(fp);
    

    free(spin_all); 
    free(zerot_avg); 
    free(fint_avg); 
    free(spin_zero); 
    free(sqt); 
    free(spint_mkl); 
    free(cmat_mkl); 
    free(qmat_mkl); 
    free(k1); 
    free(k2); 
    free(k3); 
    free(k4); 
    free(cur_input); 
    
    
    return 0;
}
