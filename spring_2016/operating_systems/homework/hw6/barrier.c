#include <stdio.h>

#include "barrier.h"

#define NUM_THREADS  (5)

struct barrier_t barrier;

void barrier_init(struct barrier_t *barrier, int count)
{
    int i;
    barrier->count = count;
    pthread_mutex_init(&barrier->mutex, NULL);
    for (i = 0; i < count; i++)
    {
        pthread_cond_init(&barrier->cond[i], NULL);
        barrier->reached[i] = 0;
    }
}

void barrier_wait(struct barrier_t *barrier, int id)
{
    // lock the mutex
    pthread_mutex_lock(&barrier->mutex);

    // mark current thread as having reached the barrier
    barrier->reached[id] = 1;

    // check if everyone arrived in the barrier
    int i;
    for (i = 0; i < barrier->count; i++)
    {
        // if not, suspend the current thread in its associated condition variable
        while (barrier->reached[i] == 0)
        {
            pthread_cond_wait(&barrier->cond[i], &barrier->mutex);
        }
    }
    
    // if so, wake everyone up and continue
    for (i = 0; i < barrier->count; i++)
        pthread_cond_signal(&barrier->cond[i]); 

    // unlock the mutex
    pthread_mutex_unlock(&barrier->mutex);
}

void* thread_func(void* arg)
{
    int tid = *((int*)arg);
    printf("Thread %d: before\n", tid);
    barrier_wait(&barrier, tid);    
    printf("Thread %d: after\n", tid);
}

int main()
{
    // init the barrier
    barrier_init(&barrier, NUM_THREADS);

    // init group of threads
    int i;
    int id[NUM_THREADS];
    pthread_t threads[NUM_THREADS];
    for (i = 0; i < NUM_THREADS; i++)
        id[i] = i;
    for (i = 0; i < NUM_THREADS; i++)
        pthread_create(&threads[i], NULL, thread_func, &id[i]);

    // join the threads
    for (i = 0; i < NUM_THREADS; i++)
        pthread_join(threads[i], NULL);

    return 0;
}
