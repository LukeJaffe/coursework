#include "command.h"
#include "terminal.h"

int main()
{
    static struct termios oldt, newt;

    /* get parameters of the current terminal
    STDIN_FILENO will tell tcgetattr that it should write the settings
    of stdin to oldt */
    tcgetattr( STDIN_FILENO, &oldt);
    
    /* now the settings will be copied */
    newt = oldt;

    /* ICANON normally takes care that one line at a time will be processed
    that means it will return if it sees a "\n" or an EOF or an EOL */
    newt.c_lflag &= ~(ICANON | ECHO);          

    /* Those new settings will be set to STDIN
    TCSANOW tells tcsetattr to change attributes immediately. */
    tcsetattr( STDIN_FILENO, TCSANOW, &newt);

    char history[MAX_HISTORY_SIZE][MAX_LINE_LEN];
    int history_len = 0;

    /* Infinite loop for input, exits on ctrl+c */
    while (1)
    {
        /* Print prompt */
        printf("$ ");
        fflush(stdout);

        /* Get line from user */
        char line[MAX_LINE_LEN];
        memset(line, 0, MAX_LINE_LEN);
        GetLine(line, history, history_len, history_len);

        /* Add this line to the command history */
        if (history_len < MAX_HISTORY_SIZE && 
                strlen(line) > 0 &&
                strcmp(line, history[history_len-1]) != 0)
        {
            strcpy(history[history_len], line);
            history_len++;
        }

        /* Parse string into subcommands and special chars */
        struct Command command; 
        ReadCommand(line, &command);
        ReadRedirectAndBackground(&command);

        /* Check if command should be in background, then run it */
        if (command.background)
            RunCommandBackground(&command);
        else
            RunCommand(&command, -1);
    }

    /* Restore the old settings */
    tcsetattr( STDIN_FILENO, TCSANOW, &oldt);

    return 0;
}
