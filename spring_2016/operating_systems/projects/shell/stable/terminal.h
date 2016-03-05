#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <termios.h>

#define clear() printf("\033[H\033[J")
#define goleft(x) printf("\033[%dD", (x))
#define goright(x) printf("\033[%dC", (x))
#define delete(x) printf("\033[%dX", (x))

#define MAX_HISTORY_SIZE    (10)
#define MAX_LINE_LEN        (200)

void GetLine(char line[MAX_LINE_LEN], char history[MAX_HISTORY_SIZE][MAX_LINE_LEN], int history_idx, int history_len);
