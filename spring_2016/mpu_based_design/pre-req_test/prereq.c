/***************************************/
/**** Name:  Luke Jaffe ****************/
/**** Email: jaffe.lu@husky.neu.edu ****/
/***************************************/

#include <stdio.h>

/* Swap elements around center pivot with xor trick */
void arrayReverse(int* array, int len)
{
    int i;
    for (i = 0; i < len/2; i++)
    {
        array[i] ^= array[len-i-1]; 
        array[len-i-1] ^= array[i]; 
        array[i] ^= array[len-i-1]; 
    }
}

void arrayPrint(int* array, int len)
{
    int i;
    printf("[ ");    
    for (i = 0; i < len; i++)
        printf("%d ", array[i]);    
    printf("]\n");    
}

unsigned char bitClear3Set7(unsigned char c)
{
    /* Bitwise AND with all ones except zero in 3rd position */
    c &= ~(1<<3);
    /* Bitwise OR with all zeroes except one in 7th position */
    c |= (1<<7);
    return c;
}

int main()
{
    /*******************************
    ** Problem 1.1: C-Programming **
    *******************************/
    /* Question 1: Where is dataArray stored?
     * Answer: dataArray is a statically allocated local array,
     * so is stored on the stack.
     *
     * Question 2: What risk does this storage location pose
     * when using larger arrays?
     * Answer: If a local array is too large, it may exceed available
     * stack space. Declaring it will cause a stack overflow. 
     */
    printf("Problem 1.1: C-Programming\n");

    int dataArray[] = {23, 12, 67, 32, 98, 19, 45, 3, 7, 72};
    int len = sizeof(dataArray)/sizeof(int);
    arrayPrint(dataArray, len);
    arrayReverse(dataArray, len);
    arrayPrint(dataArray, len);
    
    /**********************************
    ** Problem 1.2: Bit Manipulation **
    **********************************/
    printf("\nProblem 1.2: Bit Manipulation\n");

    unsigned char c = (1<<3);
    printf("0x%08x\n", c);
    c = bitClear3Set7(c);
    printf("0x%08x\n", c);

    return 0;
}
