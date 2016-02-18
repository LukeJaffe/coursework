#include <stdio.h>
#include <string.h>

int main()
{
    /* Loop until Control+C pressed */
    while (1)
    {
        /* Prompt the user */
        printf("$ ");
        /* Get the user's input */
        char buffer[50];
        fgets(buffer, 50, stdin);
        /* Remove the newline character from user's input */
        buffer[strlen(buffer)-1] = '\0';
        /* Print the user's input */
        printf("%s\n", buffer);
    }
    return 0;
}
