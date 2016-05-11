#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>

/* Fork a child process to execute the ls command
 * with its output redirected into a pipe */
void input(int* fds1, int* fds2)
{
    int ret = fork();
    if (ret < 0)
    {
        perror("fork");
        exit(1);
    }
    else if (ret == 0)
    {
        close(fds2[0]);
        close(fds2[1]);
        close(fds1[0]);
        close(1);
        dup(fds1[1]);

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

/* Fork a child process to execute the wc command
 * with its input coming from the other end of the pipe */
void output(int* fds1, int* fds2)
{
    int ret = fork();
    if (ret < 0)
    {
        perror("fork");
        exit(1);
    }
    else if (ret == 0)
    {
        close(fds1[0]);
        close(fds1[1]);
        close(fds2[1]);
        close(0);
        dup(fds2[0]);

        char* argv[3];
        argv[0] = "wc";
        argv[1] = "-l";
        argv[2] = NULL;
        execvp(argv[0], argv);
    }
    else
    {
        //nothing
    }
}

/* Fork a child process to execute the wc command
 * with its input coming from the other end of the pipe */
void middle(int* fds1, int* fds2)
{
    int ret = fork();
    if (ret < 0)
    {
        perror("fork");
        exit(1);
    }
    else if (ret == 0)
    {
        close(fds1[1]);
        close(fds2[0]);

        close(0);
        dup(fds1[0]);

        close(1);
        dup(fds2[1]);

        char* argv[3];
        argv[0] = "head";
        argv[1] = "-n 5";
        argv[2] = NULL;
        execvp(argv[0], argv);
    }
    else
    {
        //nothing
    }
}

int main()
{
    int** fds_list = malloc(2*sizeof(int));
    fds_list[0] = malloc(2*sizeof(int)); 
    fds_list[1] = malloc(2*sizeof(int)); 

    int err = pipe(fds_list[0]);
    if (err == -1)
    {
        perror("pipe");
        return 1;
    }

    err = pipe(fds_list[1]);
    if (err == -1)
    {
        perror("pipe");
        return 1;
    }

    /* Pass the pipe fds to be used by "ls" and "wc" */
    input(fds_list[0], fds_list[1]);
    middle(fds_list[0], fds_list[1]);
    output(fds_list[0], fds_list[1]);
    
    /* Parent must close these fds to avoid issues */
    close(fds_list[0][0]); 
    close(fds_list[0][1]);
    close(fds_list[1][0]); 
    close(fds_list[1][1]);

    /* Wait for all children to exit, and print message */
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
