#include <stdio.h>
#include <pthread.h>

pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
volatile int counter = 0;

void* thread_func(void* arg)
{
    int i;
    for (i = 0; i < 10000000; i++)
    {
        pthread_mutex_lock(&m);
        counter++;
        pthread_mutex_unlock(&m);
    }
}

int main()
{

    pthread_t t1, t2;
    pthread_create(&t1, NULL, thread_func, NULL);
    pthread_create(&t2, NULL, thread_func, NULL);

    pthread_join(t1, NULL);
    pthread_join(t2, NULL);

    printf("counter: %d\n", counter);
    
    return 0;
}
