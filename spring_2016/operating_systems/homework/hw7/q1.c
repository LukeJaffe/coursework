#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>

sem_t forks[5];

int left(int p)
{
    return p;
}

int right(int p)
{
    return (p + 1) % 5;
}

void getForks(int p)
{
    printf("%d tries to get forks\n", p);
    // Philosopher 0 picks up the forks in the opposite order
    // to prevent a resource dependency
    if (p == 0)
    {
        sem_wait(&forks[right(p)]);
        sem_wait(&forks[left(p)]);
    }
    else
    {
        sem_wait(&forks[left(p)]);
        sem_wait(&forks[right(p)]);
    }
    printf("%d gets the forks\n", p);
}

void putForks(int p)
{
    printf("%d puts the forks\n", p);
    sem_post(&forks[left(p)]);
    sem_post(&forks[right(p)]);
} 

void *philosopher(void *arg)
{
    int i;
    long p = (long) arg;

    // Think and eat for different amounts of time to
    // increase likelihood of deadlock (though it is 
    // not possible in this modified code)
    while (1)
    {
        // Think
        for (i = 0; i < 10000; i++);
        // Get forks
        getForks(p);
        
        // Eat
        for (i = 0; i < 100000; i++);
        // Put forks
        putForks(p);
    }
}

int main()
{
    // Initialize semaphores
    long i;
    for (i = 0; i < 5; i++)
        sem_init(&forks[i], 0, 1);
    
    // Create child threads
    pthread_t p[5];
    for (i = 0; i < 5; i++)
        pthread_create(&p[i], NULL, philosopher, (void *) i);
    
    // Wait for children
    for (i = 0; i < 5; i++)
        pthread_join(p[i], NULL);

    // Done
    return 0;
}
