#include <stdio.h>
#include <pthread.h>

#define BUFFER_LEN  (10)
#define ITERATIONS  (1000)

int buffer[BUFFER_LEN];
int head = 0, tail = 0, len = 0;
pthread_mutex_t m = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t full = PTHREAD_COND_INITIALIZER;
pthread_cond_t empty = PTHREAD_COND_INITIALIZER;

void* producer(void* arg)
{
    int i;
    for (i = 0; i < ITERATIONS; i++)
    {
        pthread_mutex_lock(&m);

        while (len == BUFFER_LEN)
           pthread_cond_wait(&full, &m); 
        buffer[head] = i;
        head = (head+1)%BUFFER_LEN;
        len++;
        pthread_cond_signal(&empty);

        pthread_mutex_unlock(&m);
    }
}

void* consumer(void* arg)
{
    int val;

    int i;
    for (i = 0; i < ITERATIONS; i++)
    {
        pthread_mutex_lock(&m);

        while (len == 0)
            pthread_cond_wait(&empty, &m);

        printf("%d\n", buffer[tail]);
        tail = (tail+1)%BUFFER_LEN;
        len--;
        pthread_cond_signal(&full);

        pthread_mutex_unlock(&m);
    }


}

int main()
{
    pthread_t pt, ct;
    pthread_create(&pt, NULL, producer, NULL);
    pthread_create(&ct, NULL, consumer, NULL);

    pthread_join(pt, NULL);
    pthread_join(ct, NULL);

    return 0;
}
