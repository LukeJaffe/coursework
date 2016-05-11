#ifndef BARRIER_H
#define BARRIER_H

#include <pthread.h>

#define BARRIER_MAX_THREADS 10

struct barrier_t
{
    int count;
    int reached[BARRIER_MAX_THREADS];
    pthread_mutex_t mutex;
    pthread_cond_t cond[BARRIER_MAX_THREADS];
};

void barrier_init(struct barrier_t *barrier, int count);

void barrier_wait(struct barrier_t *barrier, int id);

#endif
