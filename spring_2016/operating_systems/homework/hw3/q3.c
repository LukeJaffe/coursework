#include <stdio.h>
#include <stdlib.h>

void pipe_err(int err)
{
    if (err == -1)
    {
        perror("pipe");
        exit(1);
    }
}

int main()
{
    int err;
    int p2c[2];
    int c2p[2];
    pipe_err(pipe(p2c));
    pipe_err(pipe(c2p));

    int i;
    int pid = fork();
    if (pid < 0)
    {
        perror("fork");
        exit(1);
    }
    else if (pid == 0)
    {
        // Child
        char cbuf[1];
        close(c2p[0]); 
        close(p2c[1]); 
        for (i = 0; i < 5; i++)
        {
            read(p2c[0], cbuf, 1);
            printf("%d. Child\n", i + 1);
            fflush(stdout);
            write(c2p[1], " ", 1);
        }
    }
    else
    {
        // Parent
        char pbuf[1];
        close(c2p[1]); 
        close(p2c[0]); 
        for (i = 0; i < 5; i++)
        {
            printf("%d. Parent\n", i + 1);
            fflush(stdout);
            write(p2c[1], " ", 1);
            read(c2p[0], pbuf, 1);
        }
        wait(NULL);
    }
} 
