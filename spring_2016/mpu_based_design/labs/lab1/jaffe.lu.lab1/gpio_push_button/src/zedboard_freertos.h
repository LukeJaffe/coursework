#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"
#include "timers.h"
#include "xil_printf.h"
#include "xstatus.h"
#include <stdio.h>
#include "xil_types.h"
#include "xgpiops.h"
#include "xparameters.h"

#define LED_START 54
#define LED_END 61

#define SW_START 62
#define SW_END 69


#define PB_START 70
#define PB_END 74