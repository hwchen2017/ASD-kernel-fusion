#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <complex.h>
#include <time.h>
#include <unistd.h>
#include <string.h>
#include <stdbool.h>
#include <omp.h>
#include "cblas.h"
#include "v3d.h"

int N = 100; 
int num_neighbor = 4;
double  J1 = 1.0; 
int MCsweep = 500; 
double dt = 0.02; // step size in time
int Nfile = 50; 
int vx = 5, vy = 5;
int nn[4][2]; 
int trd = 1; 

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
    
    double qx, qy;
    
    qx = 2.0 * M_PI*vx/(double)N; 
    qy = 2.0 * M_PI*vy/(double)N; 
    
    
     
    double *cmat_mkl;
    cmat_mkl = (double*)malloc(N*N*N*N*sizeof(double)); 

    double *outer_mkl;
    outer_mkl = (double*)malloc(N*N*N*N*sizeof(double)); 
    
    double *spint_mkl; 
    spint_mkl = (double*)malloc(N*N*Nfile*3*sizeof(double)); 
    
    double complex *qmat_mkl; 
    qmat_mkl = (double complex*)malloc( N*N*N*N * sizeof(double complex));
    
    
    #pragma omp parallel for
    for(int j=0;j<N*N;j++)
        for(int i=0;i<N*N;i++)
        {
            int ri = i/N, ci = i%N; 
            int rj = j/N, cj = j%N; 
            double dffr = ri - rj; 
            double dffc = ci - cj;
            
            qmat_mkl[i+j*N*N] = cexp( CMPLX(0.0, -1.0) * (qx * dffr + qy * dffc ) );
        }
    
    double alpha = 1.0/(double)Nfile, beta = 0.0;
    
    double *k1, *k2, *k3, *k4, *cur_input; 
    
    k1 = (double*)malloc(N*N*Nfile*3*sizeof(double)); 
    k2 = (double*)malloc(N*N*Nfile*3*sizeof(double)); 
    k3 = (double*)malloc(N*N*Nfile*3*sizeof(double)); 
    k4 = (double*)malloc(N*N*Nfile*3*sizeof(double)); 
    cur_input = (double*)malloc(N*N*Nfile*3*sizeof(double)); 
    
    
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

            tx = tx / nrm; 
            ty = ty /nrm; 
            tz = tz /nrm; 

            spin_all[j*3] = tx,  spin_all[j*3+1] = ty, spin_all[j*3+2] = tz; 
        }



        
        t2 = omp_get_wtime();
        if(num >= 20) tlle += (t2 - t1);
        // printf("Elapsed time  of LLG:  %lf\n", t2-t1); 
        
        t1 = omp_get_wtime();
        
        #pragma omp parallel for
        for(int i=0;i<N*N;i++)
        {
            double tx=0.0, ty=0.0, tz = 0.0;

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
            for(int i=0;i<Nfile;i++)
            {
                spint_mkl[j + (i*3)*N*N] = spin_all[i*N*N*3 + j*3];
                spint_mkl[j + (i*3+1)*N*N] = spin_all[i*N*N*3 +j*3+1];
                spint_mkl[j + (i*3+2)*N*N] = spin_all[i*N*N*3 +j*3+2];
            }
        
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N*N, N*N, Nfile*3, alpha, spint_mkl, N*N, spin_zero, Nfile*3, beta, cmat_mkl, N*N); 
        

        
        #pragma omp parallel for
        for(int j=0;j<N*N;j++)
        {
            double zx = zerot_avg[j*3], zy = zerot_avg[j*3+1], zz = zerot_avg[j*3+2];
            
            for(int i=0;i<N*N;i++)
            {
                 double tx1 = fint_avg[i*3], ty1 = fint_avg[i*3+ 1], tz1 = fint_avg[i*3+ 2];
                
                 outer_mkl[i+j*N*N] = tx1*zx + ty1*zy + tz1*zz; 
            }
        }
        
        #pragma omp parallel for
        for(int i=0;i<N*N*N*N;i++)
        {
            cmat_mkl[i] -= outer_mkl[i];
        }

        
        double complex res = CMPLX(0.0, 0.0); 
        
        #pragma omp parallel
        {
            double complex res_local = CMPLX(0.0, 0.0); 
            
            #pragma omp for
            for(int i=0;i<N*N*N*N;i++)
            {
                res_local += cmat_mkl[i] * qmat_mkl[i];
            }
            
            #pragma omp critical
            {
                res += res_local; 
            }
            
        }
        
        
        res /= (double)(N*N); 
        sqt[num] = res; 
        
        
        t2 = omp_get_wtime(); 
        
        if(num >= 20) tsqt += (t2 - t1);

        // printf("Elapsed time  of MKl:  %lf\n", t3-t2);
        // cout<<"MKL Elapsed time correlation:  "<<t4-t3<<endl; 
    }
    
    tlle /= 40.0; 
    tsqt /= 40.0;
    
    
    fp = fopen("amd_gcc_11.2_gemm_openblas_sqt.txt", "a");
    fprintf(fp, "%d %lf %lf\n", N, tlle, tsqt);
    
    fclose(fp);
    
    

    free(spin_all); 
    free(zerot_avg); 
    free(fint_avg); 
    free(spin_zero); 
    free(sqt); 
    free(spint_mkl); 
    free(cmat_mkl); 
    free(outer_mkl);
    free(qmat_mkl); 
    free(k1); 
    free(k2); 
    free(k3); 
    free(k4); 
    free(cur_input); 
    

    return 0;
}
