#include <stdio.h>
#include <string.h>

void print_args(int argc, char** argv)
{
    int i;
    for (i = 0; i < argc; i++)
    {
        printf("argv[%d] = '%s'\n", i, argv[i]);
    }
}

int get_args(char* in, char** argv, int max_args)
{
    const char delim[2] = " ";
    char* token;

    /* Get first token in input */
    token = strtok(in, delim);
    argv[0] = strdup(token);

    int i;
    for (i = 1; i < max_args; i++)
    {
        token = strtok(NULL, delim);
        if (token == NULL)
            break;
        argv[i] = strdup(token);
    }

    /* Return number of arguments in argv */
    return i; 
}

int main()
{
    char s[200];
    char* argv[10];
    int argc;

    /* Read a string from the user */
    printf("Enter a string: ");
    fgets(s, sizeof s, stdin);

    /* Remove newline character from input */
    s[strlen(s)-1] = '\0';

    /* Extract arguments and print them */
    argc = get_args(s, argv, 10);
    print_args(argc, argv); 

    return 0;
}
