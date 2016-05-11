/**
 *@file main.c
 *
 *@brief Testing code for External I/O abstract
 *
 * Target:   Zedboard-Zynq-7000    
 *
 * @author    Siddharth Velu
 * @date      07/08/2010
 *
 *
 *******************************************************************************/

#include <stdio.h>
#include "gpio_counter.h"
#include "zedboard_freertos.h"

/**
 *
 * Main function that setups up GPIO and runs OS on zedboard
 *     
 *
 * Parameters:
 *
 * @return void
 */
int main( void )
{
	/* initialize GPIOs */
	gpio_init();

    /* Start the GPIO task */
    gpio_start(NULL);

    /* start the FreeRTOS scheduler
     * - will run all tasks
     * - should not terminate
     */
    vTaskStartScheduler();

    /* loop forever in case the scheduler terminates */
    for( ;; );

}
