#include <stdlib.h>
#include <fcntl.h>

#include "command.h"

#define NUM_PIPE_FDS    (2)

void RunCommand(struct Command* command, int fd);

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

/* Run a subcommand and handle piping */
int RunSubCommand(int** fds_list, int idx1, int idx2, int num_pipes, char** argv)
{
    int ret = fork();
    if (ret < 0)
    {
        perror("fork");
        exit(1);
    }
    else if (ret == 0)
    {
        int i;
        for (i = 0; i < num_pipes; i++)
        {
            if (i == idx1)
            {
                close(fds_list[i][1]);
                close(0);
                dup(fds_list[i][0]);
                close(fds_list[i][0]);
            }
            else if (i == idx2)
            {
                close(fds_list[i][0]);
                close(1);
                dup(fds_list[i][1]);
                close(fds_list[i][1]);
            }
            else
            {
                close(fds_list[i][0]);
                close(fds_list[i][1]);
            }
        }

        if (execvp(argv[0], argv) == -1)
        {
            printf("%s: Command not found\n", argv[0]);
            return;
        }
    }
    else
    {
        return ret;
    }
}

void RunCommand(struct Command* command, int fd)
{
    int i;

    /* Check if line is empty */
    if (command->num_sub_commands == 0)
    {
        return;
    }

    /* Check if line starts with <, >, & */
    if (command->sub_commands[0].argv[0] == NULL)
    {
        printf("Invalid command.\n");
        exit(1);
    }


    /* Create pipes */
    int num_pipes = command->num_sub_commands+1;
    int** fds_list = malloc(num_pipes*sizeof(int*)); 
    for (i = 0; i < num_pipes; i++)
    {
        fds_list[i] = malloc(NUM_PIPE_FDS*sizeof(int));
        if (pipe(fds_list[i]) == -1)
        {
            perror("pipe");
            exit(1);
        }
    }

    /* I/O redirection */
    int read_fd, write_fd; 
    int read_idx = -1, write_idx = -1;

    /* Check stdin redirect */
    if (command->stdin_redirect != NULL)
    {
        read_fd = open(command->stdin_redirect, O_RDONLY);
        if (read_fd != -1)
        {
            fds_list[0][0] = read_fd;
            read_idx = 0;
        }
        else
        {
            printf("%s: File not found\n", command->stdin_redirect);
            return;
        }
    }

    /* Check stdout redirect */
    if (command->stdout_redirect != NULL)
    {
        write_fd = creat(command->stdout_redirect, 0660);
        if (write_fd != -1)
        {
            fds_list[num_pipes-1][1] = write_fd;
            write_idx = num_pipes-1;
        }
        else
        {
            printf("%s: Cannot create file\n", command->stdout_redirect);
            return;
        }
    }

    /* Execute sub-commands */
    int pid;
    if (command->num_sub_commands == 1)
    {
        pid = RunSubCommand(fds_list, read_idx, write_idx, num_pipes, command->sub_commands[0].argv);
    }
    else
    {
        RunSubCommand(fds_list, read_idx, 1, num_pipes, command->sub_commands[0].argv);
        for (i = 1; i < num_pipes-2; i++)
        {
            RunSubCommand(fds_list, i, i+1, num_pipes, command->sub_commands[i].argv);
        }
        pid = RunSubCommand(fds_list, num_pipes-2, write_idx, num_pipes, command->sub_commands[num_pipes-2].argv);

    }
    
    /* If this command should be executing in the background, send pid of
     * last subcommand to parent */
    if (command->background)
    {
        char buffer[10];
        sprintf(buffer, "%d", pid);
        write(fd, buffer, sizeof(buffer));
    }

    /* Parent must close all fds to avoid issues */
    for (i = 0; i < num_pipes; i++)
    {
        close(fds_list[i][0]);
        close(fds_list[i][1]);
    }

    /* Wait for all children to exit */
    while (1)
    {
        int ret = wait(NULL);
        if (ret == -1)
            break;
    }
}

/* Run a command in the background */
void RunCommandBackground(struct Command* command)
{
    int fds[2];
    if (pipe(fds) == -1)
    {
        perror("pipe");
        exit(1);
    }

    int ret = fork();
    if (ret < 0)
    {
        perror("fork");
        exit(1);
    }
    else if (ret == 0)
    {
        RunCommand(command, fds[1]);
    }
    else
    {
        /* Parent should wait until it receives pid of last child
         * subcommand before printing next prompt */
        char buffer[10];
        read(fds[0], buffer, sizeof(buffer));
        printf("[%s]\n", buffer);
    }
}
