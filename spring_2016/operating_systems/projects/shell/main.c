#include <stdio.h>
#include <string.h>

#include "command.h"

int main()
{
    char line[200];

    while (1)
    {
        // Read a string from the user 
        printf("$ ");
        fgets(line, sizeof line, stdin);

        // Remove newline character from input 
        line[strlen(line)-1] = '\0';

        // Parse string into subcommands and print 
        struct Command command; 
        ReadCommand(line, &command);
        ReadRedirectAndBackground(&command);
        //PrintCommand(&command);
        RunCommand(&command);
    }

    return 0;
}
