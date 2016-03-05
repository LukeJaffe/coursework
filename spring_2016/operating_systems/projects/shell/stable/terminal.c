#include "terminal.h"

void GetLine(char line[MAX_LINE_LEN], char history[MAX_HISTORY_SIZE][MAX_LINE_LEN], int history_idx, int history_len)
{
    int c;   
    int cursor_idx = 0;
    int max_idx = 0;

    /* End read on newline */
    while(1)      
    {
        // Terminate on enter pressed
        c = getchar();
        if (c == '\n')
            break;

        switch (c)
        {
            case '\033':
                getchar();
                switch (getchar())
                {
                    // Up arrow key pressed
                    case 'A':
                        // decrement history_idx
                        history_idx--;
                        if (history_idx >= 0 && history_len > 0)
                        {
                            // clear line buffer
                            if (cursor_idx > 0)
                                goleft(cursor_idx);
                            delete(cursor_idx);
                            // change buffer to prev command
                            strcpy(line, history[history_idx]);
                            // print prev command to terminal
                            printf("%s", line);
                            // change cursor_idx
                            cursor_idx = (int)strlen(history[history_idx]);
                        }
                        else
                        {
                            history_idx = 0;
                        }
                        break;
                    // Down arrow key pressed
                    case 'B':
                        // increment history_idx
                        history_idx++;
                        if (history_idx < history_len && history_len > 0)
                        {
                            // clear line buffer
                            if (cursor_idx > 0)
                                goleft(cursor_idx);
                            delete(cursor_idx);
                            // change buffer to prev command
                            strcpy(line, history[history_idx]);
                            // print prev command to terminal
                            printf("%s", line);
                            // change cursor_idx
                            cursor_idx = (int)strlen(history[history_idx]);
                        }
                        else if (history_idx == history_len && history_len > 0)
                        {
                            // clear line buffer
                            if (cursor_idx > 0)
                                goleft(cursor_idx);
                            delete(cursor_idx);
                            cursor_idx = 0;
                        }
                        else
                        {
                            history_idx = history_len;
                        }
                        break;
                    // Right arrow key pressed
                    case 'C':
                        if (cursor_idx < max_idx)
                        {
                            goright(1);
                            cursor_idx++;
                        }
                        break;
                    // Left arrow key pressed
                    case 'D':
                        if (cursor_idx > 0)
                        {
                            goleft(1);
                            cursor_idx--;
                        }
                        break;
                }
                break;
            // Backspace key pressed
            case 127:
                if (cursor_idx > 0)
                {
                    goleft(1);
                    delete(1);
                    cursor_idx--;
                    max_idx--;
                }
                break;
            default:
                // Print char to terminal
                putchar(c);
                // Add char to current line
                line[cursor_idx] = c;            
                cursor_idx++;
                max_idx++;
                break;
        }
    }
    putchar('\n');
    line[cursor_idx] = '\0';
}
