/**
 *@file HelloWorld.c
 *
 *@brief
 *  - Introduction to the Zync-7000 Platform
 *  - Example of standard I/O
 * 
 * 1. Output "Hello, World!" 
 *
 *******************************************************************************/

 
#include "zedboard_freertos.h"
#include <stdio.h>

static void prvHelloWorld( void *pvParameters );

/**
 *
 * Main function to make tasks and start task scheduler*
 *     
 *
 * Parameters:
 *
 * @return void
 */
int main( void )
{
    // Hello World Task
    xTaskCreate( prvHelloWorld, ( signed char * ) "HW", configMINIMAL_STACK_SIZE, NULL, tskIDLE_PRIORITY + 3, NULL );

    /* Start the tasks and timer running. */
    vTaskStartScheduler();
	
	/** call above will never terminate, if it does, run in endless loop */
    for( ;; );
}

/**
 *
 * Sample HelloWorld task prints helloworld to STDIO 
 *     
 *
 * Parameters:
 *
 * @return void
 */
static void prvHelloWorld( void *pvParameters )
{
	printf("Input a temperature in degrees Fahrenheit\n");
	char tempf[10];
	fgets(tempf,10,stdin);
	float f = 0;
	sscanf(tempf, "%f", &f);
	float c = (f - 32.0)*(5.0/9.0);
	printf("Temperature in degrees Celsius: %f\n", c);
}
