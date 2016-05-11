#include <stdio.h>
#include <stdlib.h>

void getPWMBitPattern(unsigned int value, unsigned char patternArray[4])
{
    int shift = (value/4) - 1;
    if (value % 4 == 0)
    {
        patternArray[0] = 0x01 << shift;
        patternArray[1] = 0x01 << shift;
        patternArray[2] = 0x01 << shift;
        patternArray[3] = 0x01 << shift;
    }
    else if (value % 4 == 1)
    {
        patternArray[0] = 0x03 << shift;
        patternArray[1] = 0x01 << shift;
        patternArray[2] = 0x01 << shift;
        patternArray[3] = 0x00 << shift;
    }
    else if (value % 4 == 2)
    {
        patternArray[0] = 0x03 << shift;
        patternArray[1] = 0x03 << shift;
        patternArray[2] = 0x00 << shift;
        patternArray[3] = 0x00 << shift;
    }
    else if (value % 4 == 3)
    {
        patternArray[0] = 0x03 << shift;
        patternArray[1] = 0x02 << shift;
        patternArray[2] = 0x02 << shift;
        patternArray[3] = 0x00 << shift;
    }
}

void getPWMBitPattern2(unsigned int value, unsigned char patternArray[4])
{
    int s = (value / 4) - 1;
    unsigned int p = 2 * (value % 4);
    unsigned char r0 = 0b11111101;
    unsigned char r1 = 0b10110101;
    unsigned char r2 = 0b10000101;
    unsigned char r3 = 0b00000001;
    patternArray[0] = shift(((r0 >> p) & 0b11), s);
    patternArray[1] = shift(((r1 >> p) & 0b11), s);
    patternArray[2] = shift(((r2 >> p) & 0b11), s);
    patternArray[3] = shift(((r3 >> p) & 0b11), s);
}

unsigned char shift(unsigned char val, int mag)
{
    if (mag < 0)
        return val >> abs(mag);
    else
        return val << mag;
}

void getPWMBitPattern3(unsigned int value, unsigned char patternArray[4])
{
    int s = (value / 4) - 1;
    unsigned int p = 2 * (value % 4);
    patternArray[0] = shift(((0b11111101 >> p) & 0b11), s);
    patternArray[1] = shift(((0b10110101 >> p) & 0b11), s);
    patternArray[2] = shift(((0b10000101 >> p) & 0b11), s);
    patternArray[3] = shift(((0b00000001 >> p) & 0b11), s);
}

int main()
{
    int value = 9;
    unsigned char patternArray[4];

    int i,j;
    for (j = 0; j <= 32; j++)
    {
        printf("\nValue: %d\n", j);
        getPWMBitPattern3(j, patternArray);
        for (i = 0; i < 4; i++)
            printf("%d: %d\n", i, patternArray[i]);
    }

    return 0;
}
