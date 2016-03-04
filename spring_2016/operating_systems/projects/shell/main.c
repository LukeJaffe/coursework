#include <stdio.h>
#include <string.h>

#include "command.h"

int main()
{
    char line[200];

    /* Infinite loop for input, exits on ctrl+c */
    while (1)
    {
        /* Read a string from the user */
        printf("$ ");
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
