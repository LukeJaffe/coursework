/**
 *@file gpio.c
 *
 *@brief
 *  - Introduction lab for the Zedboard Zqnc-7000 platform
 *  - gpio skeleton code
 * 
 *
 * Target:   Zynq-7000
 *
 *
 */
/* include common definitions for running FreeRTOS on ZedBoard */
#include "zedboard_freertos.h"
#include "gpio_counter.h"

/* Priorities at which the tasks are created. */
#define taskPriority        ( tskIDLE_PRIORITY + 1 )

unsigned int* const gpio_data_ptr = GPIO_DATA;
unsigned int* const gpio_oen_ptr = GPIO_OEN;
unsigned int* const gpio_dirm_ptr = GPIO_DIRM;

static int userInput()
{
	int value;
	printf("Input a value");
	char user_input[10];
	fgets(user_input,10,stdin);
	sscanf(user_input, "%d", &value);
	return value;
}

/**
 *
 * Setups up the GPIO 
 *     
 *
 * Parameters:
 *
 * @return void
 */
void gpio_init() {
	/* Initialize pins for LEDs as output */
	*gpio_oen_ptr = 0xFF;
	*gpio_dirm_ptr = 0xFF;

	/* Initialize pins for push buttons as input  */


	/* Initialize pins for switches as input */

}


/**
 *
 * GPIO task that you create
 *
 * Parameters:
 *
 * @return void
 */
void gpio_run( void *pvParameters )
{
	const TickType_t xDelay = 500 / portTICK_PERIOD_MS;

	int user_input = userInput();
	printf("Counting to: %d\n", user_input);

    int i;
    for(i=0; i<=user_input;i++)
    {
    	*gpio_data_ptr = i;
    	vTaskDelay(xDelay);
    }
}


/**
 *
 * Runs the gpio task
 *
 * Parameters:
 *
 * @return void
 */
void gpio_start( void *pvParameters )
{
    xTaskCreate( gpio_run, ( signed char * ) "HW", configMINIMAL_STACK_SIZE, NULL, tskIDLE_PRIORITY + 1 , NULL );
}

