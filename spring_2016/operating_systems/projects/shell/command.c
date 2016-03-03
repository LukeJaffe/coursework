#include <stdlib.h>
#include <stdbool.h>

#include "command.h"

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
    if (end_index_flag)
        argv[end_index] = NULL;
}

/* Fork a child process to execute the ls command
 * with its output redirected into a pipe */
void RunSubCommand(int* fds1, int* fds2, char** argv, bool input, bool output)
{
    printf("Run sub command\n");
    // Command supplies input to a pipe and takes output from a pipe
    if (input == true && output == true)
    {
        int ret = fork();
        if (ret < 0)
        {
            perror("fork");
            exit(1);
        }
        else if (ret == 0)
        {
            // input of command is output of pipe1
            close(fds1[1]);
            close(0);
            dup(fds1[0]);
            close(fds1[0]);
            // output of command is input of pipe2
            close(fds2[0]);
            close(1);
            dup(fds2[1]);
            close(fds2[1]);
        }
        else return;
    }
    // Command supplies input to a pipe
    if (input == true && output == false)
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
            close(1);
            dup(fds2[1]);
            close(fds2[1]);
        }
        else return;
    }
    // Command takes output from a pipe
    else if (input == false && output == true)
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
            close(0);
            dup(fds1[0]);
            close(fds1[0]);
        }
        else return;
    }
    // Command does not use a pipe
    else if (input == false && output == false)
    {
        printf("No pipe\n");
        int ret = fork();
        if (ret < 0)
        {
            perror("fork");
            exit(1);
        }
        else if (ret == 0)
        {
            // Nothing
        }
        else return;
    }

    printf("Finish sub command\n");
    if (execvp(argv[0], argv) == -1)
    {
        printf("%s: Command not found\n", argv[0]);
    }
}

void RunCommand(struct Command* command)
{
    // Check if line is empty
    int i;
    if (command->num_sub_commands == 0)
    {
        printf("Empty command.\n");
        return;
    }

    // Check if line starts with <, >, & 
    if (command->sub_commands[0].argv[0] == NULL)
    {
        printf("Invalid command.\n");
        return;
    }

    // Check redirection, background
    /*
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
    */

    // create pipes 
    int num_pipes = command->num_sub_commands-1;
    int** fds_list = malloc(num_pipes*sizeof(int)); 
    for (i = 0; i < num_pipes; i++)
    {
        fds_list[i] = malloc(2*sizeof(int));
        if (pipe(fds_list[i]) == -1)
        {
            perror("pipe");
            return;
        }
    }

    if (command->num_sub_commands == 1)
    {
        printf("\nCommand without piping.\n");
        RunSubCommand(NULL, NULL, command->sub_commands[0].argv, false, false);
    }
    else
    {
        printf("\nCommand has # pipes: %d\n", num_pipes);
        RunSubCommand(NULL, fds_list[0], command->sub_commands[0].argv, true, false);
        for (i = 0; i < num_pipes-1; i++)
        {
            RunSubCommand(fds_list[i], fds_list[i+1], command->sub_commands[i+1].argv, true, true);
        }
        RunSubCommand(fds_list[num_pipes-1], NULL, command->sub_commands[num_pipes].argv, false, true);
    }

    /* Parent must close these fds to avoid issues */
    for (i = 0; i < num_pipes; i++)
    {
        close(fds_list[i][0]);
        close(fds_list[i][1]);
    }

    /* Wait for all children to exit, and print message */
    while (1)
    {
        int ret = wait(NULL);
        if (ret == -1)
            break;
        else
            printf("Process %d finished\n", ret);
    }
}
