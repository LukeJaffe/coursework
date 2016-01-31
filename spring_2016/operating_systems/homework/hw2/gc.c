#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int main()
{
    int err;

    /* Create parent-child pipe */
    int fds1[2];
    err = pipe(fds1);
    if (err == -1)
    {
        perror("pipe");
        return 1;
    }

    /* Fork 1 */
    pid_t ret1 = fork();
    if (ret1 < 0)
    {
        perror("fork");
        return 1;
    }
    else if (ret1 == 0)
    {
        /* Create child-grandchild pipe */
        int fds2[2];
        err = pipe(fds2);
        if (err == -1)
        {
            perror("pipe");
            return 1;
        }

        /* Fork 2 */
        pid_t ret2 = fork();
        if (ret2 < 0)
        {
            perror("fork");
            return 1;
        }
        else if (ret2 == 0)
        {
            // Close write end of pipe
            close(fds2[1]);

            // Duplicate read end of pipe in standard input
            close(0);
            dup(fds2[0]);

            // Child launches command "wc"
            char *argv[2];
            argv[0] = "wc";
            argv[1] = NULL;
            execvp(argv[0], argv);
        }
        else
        {
            /* Send granchild pid to parent */
            close(fds1[0]);
            char pid_buf[10];
            sprintf(pid_buf, "%d", ret2);
            write(fds1[1], pid_buf, (strlen(pid_buf) + 1));

            // Close read end of pipe
            close(fds2[0]);

            // Duplicate write end of pipe in standard output
            close(1);
            dup(fds2[1]); 

            // Parent launches command "ls -l"
            char *argv[3];
            argv[0] = "ls";
            argv[1] = "-l";
            argv[2] = NULL;
            execvp(argv[0], argv);
        }
    }
    else
    {
        /* Read grandchild pid from child */ 
        close(fds1[1]);
        char buffer[80];
        int nbytes = read(fds1[0], buffer, sizeof(buffer));
        pid_t gc_pid;
        sscanf(buffer, "%d", &gc_pid);

        /* Wait for child to finish */
        wait();
        printf("Process %d finished\n", ret1);

        /* Wait for grandchild to finish */
        waitpid(gc_pid);
        printf("Process %d finished\n", gc_pid);
    }

    return 0;
}
