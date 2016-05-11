#include <stdio.h>
#include <pthread.h>

#define DIM 5

int A[DIM * DIM];
int B[DIM * DIM];
int C[DIM * DIM];

void InitializeMatrix(int *X)
{
    int i, j;
    for (i = 0; i < DIM; i++)
    for (j = 0; j < DIM; j++)
    X[i * DIM + j] = random() % 10;
}

void PrintMatrix(int *X)
{
    int i, j;
    for (i = 0; i < DIM; i++)
    {
        for (j = 0; j < DIM; j++)
            printf("%3d ", X[i * DIM + j]);
        printf("\n");
    }
    printf("\n");
}

void MultiplyMatrices()
{
    int i, j, k;
    for (i = 0; i < DIM; i++)
    {
        for (j = 0; j < DIM; j++)
        {
            int sum = 0;
            for (k = 0; k < DIM; k++)
                sum += A[i * DIM + k] * B[k * DIM + j];
            C[i * DIM + j] = sum;
        }
    }
}

void* MultiplyMatricesThread(void* arg)
{
    int i = *((int *)arg);
    int j, k;
    for (j = 0; j < DIM; j++)
    {
        int sum = 0;
        for (k = 0; k < DIM; k++)
            sum += A[i * DIM + k] * B[k * DIM + j];
        C[i * DIM + j] = sum;
    }
}

void MultiplyMatricesParallel()
{
    int i;
    int idx[DIM];
    pthread_t threads[DIM];
    
    // generate set of indices so passing them is thread safe
    for (i = 0; i < DIM; i++)
        idx[i] = i;
    for (i = 0; i < DIM; i++)
        pthread_create(&threads[i], NULL, MultiplyMatricesThread, &idx[i]);
    for (i = 0; i < DIM; i++)
        pthread_join(threads[i], NULL);
}

int main()
{
    // initialize matrices
    InitializeMatrix(A);
    InitializeMatrix(B);
    printf("Matrix A:\n");
    PrintMatrix(A);
    printf("Matrix B:\n");
    PrintMatrix(B);

    // serial multiplication
    MultiplyMatrices();
    printf("Serial result:\n");
    PrintMatrix(C);

    // parallel multiplication
    MultiplyMatricesParallel();
    printf("Parallel result:\n");
    PrintMatrix(C);

    return 0;
}
