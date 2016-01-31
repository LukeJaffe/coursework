#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

void input(int* fds)
{
    int ret = fork();
    if (ret < 0)
    {
        perror("fork");
        exit(1);
    }
    else if (ret == 0)
    {
        close(fds[0]);
        close(1);
        dup(fds[1]);

        char* argv[3];
        argv[0] = "ls";
        argv[1] = "-l";
        argv[2] = NULL;
        execvp(argv[0], argv);
    }
    else
    {
        //nothing
    }
}

void output(int* fds)
{
    int ret = fork();
    if (ret < 0)
    {
        perror("fork");
        exit(1);
    }
    else if (ret == 0)
    {
        close(fds[1]);
        close(0);
        dup(fds[0]);

        char* argv[2];
        argv[0] = "wc";
        argv[1] = NULL;
        execvp(argv[0], argv);
    }
    else
    {
        //nothing
    }
}

int main()
{
    int fds[2];
    int err = pipe(fds);
    if (err == -1)
    {
        perror("pipe");
        return 1;
    }

    input(fds);
    output(fds);
    
    close(fds[0]); 
    close(fds[1]);

    while (1)
    {
        int ret = wait(NULL);
        if (ret == -1)
            break;
        else
            printf("Process %d finished\n", ret);
    }

    return 0;
}
