#include <stdio.h>
#include <string.h>

void print_args(int argc, char** argv)
{
    int i;
    if (argc == 0)
        printf("Input is empty or contains only spaces.\n");
    else
        for (i = 0; i < argc; i++)
        {
            printf("argv[%d] = '%s'\n", i, argv[i]);
        }
}

int get_args(char* in, char** argv, int max_args)
{
    char* token;

    /* Use " " as the token delimiter */
    const char delim[2] = " ";

    /* Get first token in input */
    token = strtok(in, delim);
    if (token == NULL)
        return 0;
    argv[0] = strdup(token);

    int i;
    for (i = 1; i < max_args; i++)
    {
        /* Get tokens from the input until none remain */
        token = strtok(NULL, delim);
        if (token == NULL)
            break;

        /* Allocate new memory and duplicate the token */
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
