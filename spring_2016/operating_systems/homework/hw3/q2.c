#include <stdio.h>
#include <string.h>

#define MAX_SUB_COMMANDS    (5)
#define MAX_ARGS            (10)

struct SubCommand
{
    char* line;
    char* argv[MAX_ARGS];
};

struct Command
{
    struct SubCommand sub_commands[MAX_SUB_COMMANDS];
    int num_sub_commands;
};

void PrintArgs(char** argv)
{
    int i = 0;
    while (argv[i] != NULL)
    {
        printf("argv[%d] = '%s'\n", i, argv[i]);
        i++;
    }
}

void ReadArgs(char* in, char** argv, int size)
{
    char* token;

    /* Use " " as the token delimiter */
    const char delim[2] = " ";

    /* Get first token in input */
    token = strtok(in, delim);
    if (token == NULL)
    {
        argv[0] = NULL;
        return;
    }
    argv[0] = strdup(token);

    int i;
    for (i = 1; i < size-1; i++)
    {
        /* Get tokens from the input until none remain */
        token = strtok(NULL, delim);
        if (token == NULL)
            break;

        /* Allocate new memory and duplicate the token */
        argv[i] = strdup(token);
    }

    argv[i] = NULL; 
}

void ReadCommand(char* line, struct Command* command)
{
    char* token;

    /* Use "|" as the token delimiter */
    const char delim[2] = "|";

    /* Get first token in input */
    token = strtok(line, delim);
    if (token == NULL)
    {
        command->num_sub_commands = 0;
        return;
    }
    command->sub_commands[0].line = strdup(token);

    int i;
    for (i = 1; i < MAX_SUB_COMMANDS; i++)
    {
        /* Get tokens from the input until none remain */
        token = strtok(NULL, delim);
        if (token == NULL)
            break;

        /* Allocate new memory and duplicate the token */
        command->sub_commands[i].line = strdup(token);
    }

    command->num_sub_commands = i;

    for (i = 0; i < command->num_sub_commands; i++)
    {
        /* Process the subcommand */
        ReadArgs(command->sub_commands[i].line,
                command->sub_commands[i].argv,
                MAX_ARGS);
    }

}

void PrintCommand(struct Command* command)
{
    int i;
    for (i = 0; i < command->num_sub_commands; i++)
    {
        printf("\nSubcommand %d:\n", i+1);
        PrintArgs(command->sub_commands[i].argv);
    }
}

int main()
{
    char line[200];

    /* Read a string from the user */
    printf("Enter a string: ");
    fgets(line, sizeof line, stdin);

    /* Remove newline character from input */
    line[strlen(line)-1] = '\0';

    /* Parse string into subcommands and print */
    struct Command command; 
    ReadCommand(line, &command);
    PrintCommand(&command);

    return 0;
}
