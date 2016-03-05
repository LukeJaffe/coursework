#include <string.h>
#include <stdio.h>

#include "command.h"

#define MAX_LINE_LEN        (200)

int main()
{
    /* Infinite loop for input, exits on ctrl+c */
    while (1)
    {
        char line[MAX_LINE_LEN];
        /* Print prompt */
        printf("$ ");

        /* Read a string from the user */
        fgets(line, sizeof line, stdin);

        /* Remove newline character from input */
        line[strlen(line)-1] = '\0';

        /* Parse string into subcommands and special chars */
        struct Command command; 
        ReadCommand(line, &command);
        ReadRedirectAndBackground(&command);

        /* Check if command should be in background, then run it */
        if (command.background)
            RunCommandBackground(&command);
        else
            RunCommand(&command, -1);
    }

    return 0;
}
