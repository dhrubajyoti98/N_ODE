/*
A N-th order ODE Solver.
Author: Dhrubajyoti Biswas, Date: 30th July 2019.

The RHS of the N- first order systems has to be passed as function pointers.
 */
#include<cstdio>
#include<cstdlib>
#include<cmath>
#include<omp.h>


class N_ODE
{
    private:
    int N;
    double h;
    int CORE;

    double beta(int j)
    {
        if(j==0)
            return 0;
        else if(j==3)
            return 1;
        else
            return 0.5;
    }

    double alpha(int l, int j, double K[][4])
    {
        if(j==0)
            return 0;
        else if(j==3)
            return K[l][j-1];
        else
            return (0.5)*K[l][j-1];
    }

    void modify(double a[],double K[][4], int j, int del)
    {
        int l;
        #pragma omp parallel for num_threads(1)
        for(l=0;l<N;l++)
        {
            a[l]=a[l]+del*alpha(l,j,K);
        }
    }

    public:
    double t;

    N_ODE(int NN,double hh)
    {
        N=NN;
        t=0;
        h=hh;
    }

    void solveRK4(double x_now[], double x_copy[], double (*f[])(double[],double), double WORKSPACE[][4], int CORE=-1)
    {
        CORE=(CORE==-1)?N:CORE;
        double t_c;
        
        for(int j=0;j<4;j++)
        {
            t_c=t+beta(j)*h;
            
            modify(x_copy, WORKSPACE, j,1);

            #pragma omp parallel for num_threads(CORE)
            for(int i=0;i<N;i++)
            {
                WORKSPACE[i][j]=h*(*f[i])(x_copy,t_c);
            }

            modify(x_copy, WORKSPACE, j, -1);

        }

        #pragma omp parallel for num_threads(CORE)
        for(int i=0;i<N;i++)
        {
            x_now[i]=x_now[i]+(1/6.0)*(WORKSPACE[i][0]+WORKSPACE[i][3]+2*(WORKSPACE[i][1]+WORKSPACE[i][2]));
        }
        t=t+h;
    }
};

