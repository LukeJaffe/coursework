/**
 *@file gpio.h
 *
 *@brief
 *  - GPIO Skeleton
 *
 * Target:   Zedboard
 *
 *
 *******************************************************************************/
#ifndef GPIO_H_
#define GPIO_H_
/** gpio_init
 *
 * Initialization of PORTFIO. This PORT is used as GPIO.
 * The output and input direction can be set using the MACROS
 *
 * Parameters:
 *
 * @return void
 */
void gpio_init(void);

/** gpio_init
 *
 * The main command loop. Write all the control commands in this function
 *
 * Parameters:
 *
 * @return void
 */
void gpio_start(void *pvParameters);

/* Insert definitions for MMR addresses for GPIO here
 * first define the BASE, and then offset for each MMR.
 * See the following examples below. Uncomment, fill in the
 * necessary addresses, and add more definitions as needed */

void toggle_led();

// /** Base address for GPIO peripheral*/
 #define GPIO_BASE	(0xE000A000)

/// /** Output Data (GPIO Bank2, EMIO) */
#define GPIO_DATA (GPIO_BASE + 0x00000048)
#define GPIO_OEN (GPIO_BASE + 0x00000288)
#define GPIO_DIRM (GPIO_BASE + 0x00000284)



#endif /* GPIO_H_ */


