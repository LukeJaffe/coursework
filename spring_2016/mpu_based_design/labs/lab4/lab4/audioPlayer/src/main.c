
#include "zedboard_freertos.h"
#include "audioPlayer.h"

audioPlayer_t audioPlayer;

int main()
{
	// I2C and I2S initialization.
	audioPlayer_init(&audioPlayer);

	// Create the Audio player task.
	audioPlayer_start(&audioPlayer);

	// start the OS scheduler to kick off the tasks.
	vTaskStartScheduler();
	return(0);
}
