#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main()
{
    int ret = fork();

    /* Fork error */
    if (ret < 0)
    {
        printf("Fork Error\n");
        exit(1);
    }
    /* Child process */
    else if (ret == 0)
    {
        int input;
        char buffer[10];
        printf("Enter a number: ");
        fgets(buffer, 10, stdin);
        sscanf(buffer, "%d", &input);
        return input;
    }
    /* Parent process */
    else
    {
        /* Parent waits for child */
        int input;
        wait(&input);
        /* Max value preserved is 255 because WEXISTATUS,
         * "returns the exit status of the child.  
         * This consists of the least significant 8 bits
         * of the status argument that the child specified".
         * The max value which can be contained in 8 bits is 255 (base 10).*/
        printf("Child exited with status %d\n", WEXITSTATUS(input)); 
    }

    return 0;
}
