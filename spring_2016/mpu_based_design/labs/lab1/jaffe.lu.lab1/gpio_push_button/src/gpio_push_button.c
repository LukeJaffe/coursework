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
#include "gpio_push_button.h"

/* Priorities at which the tasks are created. */
#define taskPriority        ( tskIDLE_PRIORITY + 1 )

unsigned int* const gpio_data_ptr = GPIO_DATA;
unsigned int* const gpio_oen_ptr = GPIO_OEN;
unsigned int* const gpio_dirm_ptr = GPIO_DIRM;
static unsigned int led_state = 0;

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
	*gpio_oen_ptr = 0x000000FF;
	*gpio_dirm_ptr = 0x000000FF;

	/* Initialize pins for push buttons as input  */


	/* Initialize pins for switches as input */

}

void toggle_led()
{
	if (led_state)
		*gpio_data_ptr = (0xFF);
	else
		*gpio_data_ptr = (0x00);
	led_state = led_state ? 0 : 1;
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
    for( ;; )
    {
    	int i;
    	for (i = 0; i < 8; i++)
    	{
    		int j = i + 8;
    		int led_bit = (*gpio_data_ptr) & (0x01<<i);
    		int switch_bit = (*gpio_data_ptr) & (0x01<<j);
    		switch_bit = switch_bit >> 8;
    		if (led_bit ^ switch_bit)
    		{
    			if (switch_bit)
    				printf("LED%d is now ON\n", i);
    			else
    				printf("LED%d is now OFF\n", i);
    		}
    	}
    	(*gpio_data_ptr) = (((*gpio_data_ptr)>>8) & 0xFF);
    	/* Wait for ... seconds */
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

