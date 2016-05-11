#include <stdio.h>
#include <stdlib.h>

unsigned char shift(unsigned char val, int mag)
{
    if (mag < 0)
        return val >> abs(mag);
    else
        return val << mag;
}

void getPWMChangeBitPattern(unsigned int value, unsigned int patternArray[4])
{
    int s = (value / 4) - 1;
    unsigned int p = 2 * (value % 4);

    unsigned int d0 = shift(((0b11111101 >> p) & 0b11), s);
    unsigned int d1 = shift(((0b10110101 >> p) & 0b11), s);
    unsigned int d2 = shift(((0b10000101 >> p) & 0b11), s);
    unsigned int d3 = shift(((0b00000001 >> p) & 0b11), s);

    // assume all LEDs are off at the beginning of a period
    unsigned int m0 = (~d0 << 16) & 0xFFFF0000; 
    unsigned int m1 = (~(d1 ^ d0) << 16) & 0xFFFF0000;
    unsigned int m2 = (~(d2 ^ d1) << 16) & 0xFFFF0000;
    unsigned int m3 = (~(d3 ^ d2) << 16) & 0xFFFF0000;

    patternArray[0] = m0 | d0;
    patternArray[1] = m1 | d1;
    patternArray[2] = m2 | d2;
    patternArray[3] = m3 | d3;
}

int main()
{
    unsigned int value = 9;
    unsigned int patternArray[4];

    int i,j;
    for (j = 0; j <= 32; j++)
    {
        printf("\nValue: %d\n", j);
        getPWMChangeBitPattern(j, patternArray);
        for (i = 0; i < 4; i++)
            printf("%d: %x\n", i, patternArray[i]);
    }

    return 0;
}
