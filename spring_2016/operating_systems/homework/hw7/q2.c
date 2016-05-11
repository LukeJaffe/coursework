#include <stdio.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_ROWS    (4)
#define NUM_COLS    (4)

#define NUM_SENDERS (2)

pthread_mutex_t M[NUM_ROWS][NUM_COLS];

void* sender(void* arg)
{
    // Which packet is this thread
    long s = (long)arg;

    // Send packets through the network forever
    int i;
    while (1)
    {
        // Calculate a random src node
        int src[] = {random()%NUM_COLS, random()%NUM_ROWS};

        // Calculate a random dst node, different from src node
        int dst[2];
        do
        {
            dst[0] = random()%NUM_COLS; 
            dst[1] = random()%NUM_ROWS;
        }
        while(src[0] == dst[0] && src[1] == dst[1]);

        // Packet initially occupies position of src node
        int pos[] = {src[0], src[1]};

        // Temp variables for new position
        int x=src[0], y=src[1];

        // Acquire the src position
        pthread_mutex_lock(&M[pos[0]][pos[1]]);

        // While the packet has not reached dst x position
        while (pos[0] != dst[0])
        {
            // Calculate the new x position
            if (pos[0] > dst[0])
                x--;
            else
                x++;

            // Print current and next position
            printf("packet: %ld, pos: (%d, %d), next: (%d, %d)\n", s, pos[0], pos[1], x, pos[1]);

            // Acquire the new position
            pthread_mutex_lock(&M[x][pos[1]]);

            // Release the old position
            pthread_mutex_unlock(&M[pos[0]][pos[1]]);

            // Set the current x to the new x
            pos[0] = x;
        }
        
        // While the packet has not reached dst y position
        while (pos[1] != dst[1])
        {
            // Calculate the new y position
            if (pos[1] > dst[1])
                y--;
            else
                y++;

            // Print current and next position
            printf("packet: %ld, pos: (%d, %d), next: (%d, %d)\n", s, pos[0], pos[1], pos[0], y);

            // Acquire the new position
            pthread_mutex_lock(&M[pos[0]][y]);

            // Release the old position
            pthread_mutex_unlock(&M[pos[0]][pos[1]]);

            // Set the current y to the new y
            pos[1] = y;
        }
        
        // Release the dst position
        pthread_mutex_unlock(&M[pos[0]][pos[1]]);
    }
}

int main()
{
    // Set up network of mutexes
    long i, j;
    for (i = 0; i < NUM_COLS; i++)
        for (j = 0; j < NUM_ROWS; j++)
            pthread_mutex_init(&M[i][j], NULL);

    // Array of sender threads
    pthread_t s[NUM_SENDERS];

    // Create threads
    for (i = 0; i < NUM_SENDERS; i++)
        pthread_create(&s[i], NULL, sender, (void*)i);

    // Wait for threads
    for (i = 0; i < NUM_SENDERS; i++)
        pthread_join(s[i], NULL);

    return 0;
}
