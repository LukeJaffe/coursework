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

void PrintArgs(char** argv);
void ReadArgs(char* in, char** argv, int size);
void ReadCommand(char* line, struct Command* command);
void PrintCommand(struct Command* command);
void ReadRedirectAndBackground(struct Command* command);
