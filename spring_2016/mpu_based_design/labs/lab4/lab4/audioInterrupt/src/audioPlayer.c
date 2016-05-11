#include "audioPlayer.h"
#include "adau1761.h"
#include "zedboard_freertos.h"
#include "gpio_interrupt.h"
#include <stdio.h>

#define VOLUME_CHANGE_STEP (4)
#define VOLUME_MAX (0xEF)
#define VOLUME_MIN (0x2F)

extern unsigned int  snd_samples_nSamples;
extern unsigned int snd_samples[];

static int gpio_setupInts(audioPlayer_t *pThis);
static void gpio_intrHandler(void *pRef);

// Initialize the audio player and everything it contains
int audioPlayer_init(audioPlayer_t *pThis)
{
	//Fill the audioPlayer instance with default parameters.

	pThis->volume 		= VOLUME_MAX; /*default volume */
	pThis->frequency 	= 16000; /* default frequency */
	pThis->AXI_I2S_RATE = 48000;

	adau1761_init(&(pThis->codec));

	return 0;
}



void audioPlayer_volumeIncrease(audioPlayer_t *pThis)
{
	/* insert code for volume increase here */
	/* for this first, implement adau1761_setVolume in adau1761.c */
	if (pThis->volume < VOLUME_MAX)
		pThis->volume += 4;
	adau1761_setVolume(&(pThis->codec), pThis->volume);
}

void audioPlayer_volumeDecrease(audioPlayer_t *pThis)
{
	/* insert code for volume decrease here */
	if (pThis->volume > 0)
		pThis->volume -= 4;
	adau1761_setVolume(&(pThis->codec), pThis->volume);
}


/*
 * This solution is set for MONO Audio O/P only.
 */
void txData(unsigned int* sampleA, int len)
{
	int i = 0;
	while (i < len)
	{
		if (*(volatile u32 *) (FIFO_BASE_ADDR + FIFO_TX_VAC))
		{
			u32 sample_32 = (sampleA[i] << 16) & 0xFFFF0000;
			*(volatile u32 *) (FIFO_BASE_ADDR + FIFO_TX_DATA) = sample_32;
			*(volatile u32 *) (FIFO_BASE_ADDR + FIFO_TX_LENGTH) = 1;
			*(volatile u32 *) (FIFO_BASE_ADDR + FIFO_TX_DATA) = sample_32;
			*(volatile u32 *) (FIFO_BASE_ADDR + FIFO_TX_LENGTH) = 1;
			i++;
		}
		else
		{
			vTaskDelay(1); //delay for 0-10 ms
		}
	}
}

static void audioPlayer_task( void *pThis ){
	// insert code for outputting samples to FIFO Q here.
	// make sure to check if there is space in the FIFO before writing to it.
	// after setting an writing an sample, write the packet length of 1 as well
	// AXI Streaming FIFO expects packet information
	//
	// Audio samples are stored in:   snd_samples[] as 16 bit mono
	gpio_init();
	gpio_setupInts(pThis);
	audioPlayer_init(pThis);
	while (1)
		txData(snd_samples, snd_samples_nSamples);
}

int audioPlayer_start(audioPlayer_t *pThis) {
	/* create the audio task */
	printf("Start!\n");
	xTaskCreate( audioPlayer_task, ( signed char * ) "HW", configMINIMAL_STACK_SIZE, pThis, tskIDLE_PRIORITY + 1 , NULL );
	return 0;
}

static int gpio_setupInts(audioPlayer_t *pThis) {
	// pointer to driver structure
	XScuGic *pGIC;
	// get pointer to GIC (already initialized at OS startup
	pGIC = prvGetInterruptControllerInstance();
	// connect own interrupt handler to GIC handler
	XScuGic_Connect(pGIC, GPIO_INTERRUPT_ID,
	(Xil_ExceptionHandler) gpio_intrHandler,(void *) pThis);
	// Enable interrupt at GIC
	XScuGic_Enable(pGIC, GPIO_INTERRUPT_ID);
	/* Enable IRQ at core (should be enabled anyway)*/
	Xil_ExceptionEnableMask(XIL_EXCEPTION_IRQ);

	/* Enable IRQ in processor core  */

	return XST_SUCCESS;
}

static void gpio_intrHandler(void *pRef)
{
	// read interrupt status
	u32 int_assert = (*(volatile u32 *)GPIO_INT_STAT_2) & ~(*(volatile u32 *)GPIO_INT_MASK_2);

	// clear interrupts
	(*(volatile u32 *)GPIO_INT_STAT_2) = int_assert;


	/* counter manipulation */
	/* Send the counter value to Queue */
	if (int_assert & (0x01<<20))
	{
		audioPlayer_volumeIncrease((audioPlayer_t *)pRef);
	}
	if (int_assert & (0x01<<18))
	{
		audioPlayer_volumeDecrease((audioPlayer_t *)pRef);
	}

	unsigned int n;
	for ( n=0; n < 5000000; n++ )
	{
		asm("nop;");
	}
}

void gpio_init(void) {

    /* OutEnable for LEDs which is top 8 bits need to be set to 1 */
	*(volatile u32 *)GPIO_DIRM_2 = 0x00000000;
    *(volatile u32 *)GPIO_OEN_2 =  0x00000000;

    /* disable interrupts before configuring new ints */
    *(volatile u32 *)GPIO_INT_DIS_2 = 0xFFFFFFFF;

    *(volatile u32 *)GPIO_INT_TYPE_2 = (0x01<<18 | 0x01<<20);
    *(volatile u32 *)GPIO_INT_POLARITY_2 = (0x01<<18 | 0x01<<20);
    *(volatile u32 *)GPIO_INT_ANY_2 = 0x00000000;

    /* enable input bits */
    *(volatile u32 *)GPIO_INT_EN_2 = (0x01<<18 | 0x01<<20);
}
