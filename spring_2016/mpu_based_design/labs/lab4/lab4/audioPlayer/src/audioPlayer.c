#include "audioPlayer.h"
#include "adau1761.h"
#include "zedboard_freertos.h"
#include <stdio.h>

#define VOLUME_CHANGE_STEP (4)
#define VOLUME_MAX (0xEF)
#define VOLUME_MIN (0x2F)

extern unsigned int  snd_samples_nSamples;
extern unsigned int snd_samples[];

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

}

void audioPlayer_volumeDecrease(audioPlayer_t *pThis)
{
	/* insert code for volume decrease here */
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
