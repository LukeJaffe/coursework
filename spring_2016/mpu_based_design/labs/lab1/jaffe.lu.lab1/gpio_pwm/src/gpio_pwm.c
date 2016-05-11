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
#include "gpio_pwm.h"

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
	*gpio_oen_ptr = 0xFF;
	*gpio_dirm_ptr = 0xFF;

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
    	printf("test\n");
        /* insert your code to manipulate the GPIOs here */
    	/* Toggle all LEDs */
    	toggle_led();
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

