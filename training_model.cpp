#include <iostream>
#include <cmath>
#include <fstream>
#include<time.h>
using namespace std;

float log_s(float x)
{
    float y = 1 / (1 + exp(-x));
    return y;
}
int main()
{
    ifstream in;
    ofstream out;
    int i, j, k, p;
    int L, M, N, P;

    L = 5;
    M = 9;
    N = 3;
    P = 12;

    float I[L][P], V[L + 1][M],delv[L+1][M] ,W[M + 1][N],delw[M+1][N], IH[M][P], OH[M][P],IO[N][P], OO[N][P], T[N][P],err,error,MSE;
    float  E[N][P], e[k][P], sum, c, nita = 0.6,alpha=0.001, Imax, Imin, Tmax, Tmin, Omax, Omin;
    // input for the input layer//
    in.open("input_data_training.txt");
    for (i = 0; i < L; i++)
    {
        for (p = 0; p < P; p++)
        {
            in >> I[i][p];
        }
    }
    in.close();

    // target value for the output layer//
    cout << endl;
    in.open("target_data_training.txt");
    for (k = 0; k < N; k++)
    {
        for (p = 0; p < P; p++)
        {
            in >> T[k][p];
        }
    }
    in.close();

    // normalizing the input to input layer//

    for (i = 0; i < L; i++)
    {

        Imax = I[i][0];
        Imin = I[i][0];
        for (p = 0; p < P; p++)
        {
            if (I[i][p] > Imax)
            {
                Imax = I[i][p];
            }
            if (I[i][p] < Imin)
            {
                Imin = I[i][p];
            }
        }

        for (p = 0; p < P; p++)
        {
            I[i][p] = 0.1 + 0.8 * ((I[i][p] - Imin) / (Imax - Imin));
        }
    }

    cout << "input after normalization:" << endl;
    for (i = 0; i < L; i++)
    {
        for (p = 0; p < P; p++)
        {
            cout << I[i][p] << " ";
        }
        cout << endl;
    }

    // finding the max and min target value//
    for (k = 0; k < N; k++)
    {

        Tmax = T[k][0];
        Tmin = T[k][0];
        for (p = 0; p < P; p++)
        {
            if (T[k][p] > Tmax)
            {
                Tmax = T[k][p];
            }
            if (T[k][p] < Tmin)
            {
                Tmin = T[k][p];
            }
        }

        for (p = 0; p < P; p++)
        {
            T[k][p] = 0.1 + 0.8 * ((T[k][p] - Tmin) / (Tmax - Tmin));
        }
    }

    cout << endl;
    cout << "target value after normalizing:" << endl;

    for ( k = 0; k < N; k++)
    {
        for ( p = 0; p < P; p++)
        {
            cout<<T[k][p]<<" ";
        }
        cout<<endl;
        
    }
    

    // initial input to connection weight bw input and hidden layer//

    for (i = 0; i < L + 1; i++)
    {
        for (j = 0; j < M; j++)
        {
            V[i][j] = sin(rand());
        }
    }

    // initial input to connection weight bw hidden and output layer//
    for (j = 0; j < (M + 1); j++)
    {
        for (k = 0; k < N; k++)
        {
            W[j][k] = sin(rand());
        }
    }

    // input for the Hidden neuron//
    int itr = 1;
    MSE=0.1;
    while (itr < 5000000 && MSE > 0.00001)
    {

        for (p = 0; p < P; p++)
        {
            for (j = 0; j < M; j++)
            {
                sum = 0;
                IH[j][p] = 0;
                for (i = 0; i < L; i++)
                {
                    sum = sum + I[i][p] * V[i][j];
                }
                IH[j][p] = sum + 1 * V[L][j];
            }
        }

        // outpot for Hidden neuron//
        for (p = 0; p < P; p++)
        {
            for (j = 0; j < M; j++)
            {
                OH[j][p] = log_s(IH[j][p]);
            }
        }

        // input for the output neuron//

        for (p = 0; p < P; p++)
        {
            for (k = 0; k < N; k++)
            {
                sum = 0;
                IO[k][p] = 0;
                for (j = 0; j < M; j++)
                {
                    sum = sum + OH[j][p] * W[j][k];
                }
                IO[k][p] = sum + 1 * W[M + 1][k];
            }
        }
        // outpot for output neuron//
        for (p = 0; p < P; p++)
        {
            for (k = 0; k < N; k++)
            {
                OO[k][p] = log_s(IO[k][p]);
            }
        }
       

        // error for the output layer//
        for (p = 0; p < P; p++)
        {
            for (k = 0; k < N; k++)
            {
                E[k][p] = T[k][p] - OO[k][p];
                e[k][p] = 0.5 * pow(E[k][p], 2); // energy of error signal//
            }
        }

        // now we have to update W[j][k] and V[i][j] for this we have calculated the formula that we will use here
        // here "nita" is the learning rate ...

        // W(new)=W(old)+nita*del(W)
        // V(new)=V(old)+nita*del(V)

        // updating "W" weight//
        for (p = 0; p < P; p++)
        {
            for (j = 0; j < (M + 1); j++)
            {
                for (k = 0; k < N; k++)
                {
                    delw[j][k]= nita * (T[k][p] - OO[k][p]) * OO[k][p] * (1 - OO[k][p]) * OH[j][p];
                    if(itr==1)
                    W[j][k] = W[j][k]+delw[j][k];
                    else
                    W[j][k] = W[j][k]+delw[j][k]+alpha*delw[j-1][k-1];
                }
            } 
        }

        for (p = 0; p < P; p++)
        {
            for (i = 0; i < L + 1; i++)
            {
                for (j = 0; j < M; j++)
                {
                    sum = 0;
                    for (k = 0; k < N; k++)
                    {
                        sum = sum + ((T[k][p] - OO[k][p]) * OO[k][p] * (1 - OO[k][p]) * W[j][k] * OH[j][p] * (1 - OH[j][p]) * I[i][p]);
                    }
                    delv[i][j]=(nita/N)*sum;
                    V[i][j] = V[i][j] +delv[i][j];

                }
            }
        }

         error = 0;
        for (p = 0; p < P; p++)
        {
            sum = 0;
            for (k = 0; k < N; k++)
            {
                sum = sum + e[k][p];
            }
            err = sum / N;
            error = error + err;
        }

        MSE = error / P;

        itr++;
    }

    cout << "MSE is :" << MSE << endl;
    cout<<"iteration is:"<<itr<<endl;
    out.open("final_v.txt");
    for (i = 0; i < L + 1; i++)
    {
        for (j = 0; j < M; j++)
        {
            out << V[i][j] << " ";
        }
        out << endl;
    }
    out.close();

    out.open("final_w.txt");
    for (j = 0; j < (M + 1); j++)
    {
        for (k = 0; k < N; k++)
        {
            out << W[j][k] << " ";
        }
        out << endl;
    }
    out.close();

    // NOW THE MODEL IS TRAINED SO ,NOW SATRTING THE TESTING OF MODEL//

    return 0;
}