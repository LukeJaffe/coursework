#ifndef __AUDIOPLAYER_H_
#define __AUDIOPLAYER_H_

#include "xparameters.h"
#include "adau1761.h"

/* Audio Player Structure */
typedef struct{
	unsigned int volume;
	unsigned int frequency;
	unsigned int AXI_I2S_RATE;
	tAdau1761 codec;
}audioPlayer_t;


/* Externally available functions */

/** initialize audio player
 *@param pThis  pointer to own object
 *
 *@return 0 success, non-zero otherwise
 **/
int audioPlayer_init(audioPlayer_t *pThis);

/** startup phase after initialization
 *@param pThis  pointer to own object
 *
 *@return 0 success, non-zero otherwise
 **/
int audioPlayer_start(audioPlayer_t *pThis);

#endif
