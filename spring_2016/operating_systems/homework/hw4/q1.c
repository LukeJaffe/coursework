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
    char* stdin_redirect;
    char* stdout_redirect;
    int background;
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

    /* Check if line is empty */
    if (command->num_sub_commands == 0)
    {
        printf("Empty command.\n");
        return;
    }
     
    /* Check if line starts with <, >, & */
    if (command->sub_commands[0].argv[0] == NULL)
    {
        printf("Invalid command.\n");
        return;
    }

    int i;
    for (i = 0; i < command->num_sub_commands; i++)
    {
        printf("\nCommand %d:\n", i);
        PrintArgs(command->sub_commands[i].argv);
    }

    if (command->stdin_redirect == NULL)
        printf("\nRedirect stdin: NULL\n");
    else
        printf("\nRedirect stdin: %s\n", command->stdin_redirect);

    if (command->stdout_redirect == NULL)
        printf("Redirect stdout: NULL\n");
    else
        printf("Redirect stdout: %s\n", command->stdout_redirect);

    if (command->background)
        printf("Background: yes\n");
    else
        printf("Background: no\n");
}

/* This function does not handle all possible corner cases, 
 * but correctly performs all tasks in the problem statement */
void ReadRedirectAndBackground(struct Command* command)
{
    const char left_carrot[2] = "<";
    const char right_carrot[2] = ">";
    const char ampersand[2] = "&";
    
    unsigned int left_carrot_flag = 0;
    unsigned int right_carrot_flag = 0;

    unsigned int end_index_flag = 0;
    unsigned int end_index = 0;

    char** argv; 

    /* Default redirection, background fields */
    command->stdin_redirect = NULL;
    command->stdout_redirect = NULL;
    command->background = 0;

    /* Check if empty line */
    if (command->num_sub_commands == 0)
        return;
    else
        argv = command->sub_commands[command->num_sub_commands-1].argv;

    /* Iterate through all tokens in last subcommand */
    int i = 0;
    while (argv[i] != NULL)
    {
        if (left_carrot_flag)
        {
            /* If previous token was <, this token is file */
            command->stdin_redirect = strdup(argv[i]);
            left_carrot_flag = 0;     
        }
        else if (right_carrot_flag)
        {
            /* If previous token was >, this token is file */
            command->stdout_redirect = strdup(argv[i]);
            right_carrot_flag = 0;     
        }
        else
        {
            /* Check for <, >, and set flag to indicate if found.
             * Mark where first special character is found. */
            if (strcmp(argv[i], left_carrot) == 0)
            {
                left_carrot_flag = 1;
                if (!end_index_flag)
                {
                    end_index_flag = 1;
                    end_index = i;
                }
            }
            else if (strcmp(argv[i], right_carrot) == 0)
            {
                right_carrot_flag = 1;
                if (!end_index_flag)
                {
                    end_index_flag = 1;
                    end_index = i;
                }
            }
            else if (strcmp(argv[i], ampersand) == 0)
            {
                if (!end_index_flag)
                {
                    end_index_flag = 1;
                    end_index = i;
                }
            }
        }
        i++;
    }

    /* Check if last character is & */
    if (i > 0) 
    {
        if (strcmp(argv[i-1], ampersand) == 0)
            command->background = 1;
        else
            command->background = 0;
    }

    /* NULL terminate the final subcommand where special characters begin */
    argv[end_index] = NULL;
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
    ReadRedirectAndBackground(&command);
    PrintCommand(&command);

    return 0;
}
