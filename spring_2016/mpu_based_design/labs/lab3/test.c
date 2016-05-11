#include <stdio.h>

int foo()
{
    static int x = 0;
    return x++;
}

int main()
{
    int i;
    for (i = 0; i < 10; i++)
        printf("%d\n", foo());
    return 0;
}
