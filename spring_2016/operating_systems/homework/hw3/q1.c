#include <stdio.h>
#include <string.h>

#define MAX_ARGS    (10)

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


    /* Iterate to size-1 to save a spot in array for NULL */
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

    /* NULL terminate the array */
    argv[i] = NULL; 
}

int main()
{
    char s[200];

    char* argv[MAX_ARGS];
    int argc;

    /* Read a string from the user */
    printf("Enter a string: ");
    fgets(s, sizeof s, stdin);

    /* Remove newline character from input */
    s[strlen(s)-1] = '\0';

    /* Extract arguments and print them */
    ReadArgs(s, argv, MAX_ARGS);
    PrintArgs(argv); 

    return 0;
}
