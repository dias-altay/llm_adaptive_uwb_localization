/*
 * Copyright (c) 2015 Nordic Semiconductor. All Rights Reserved.
 * Adapted based on https://github.com/TIERS/dwm1001-uwb-firmware/ open-source firmware.
 */

#include "sdk_config.h"
#include "FreeRTOS.h"
#include "task.h"
#include "timers.h"
#include "bsp.h"
#include "boards.h"
#include "nordic_common.h"
#include "nrf_drv_clock.h"
#include "nrf_drv_spi.h"
#include "nrf_uart.h"
#include "app_util_platform.h"
#include "nrf_gpio.h"
#include "nrf_delay.h"
#include "nrf_log.h"
#include "nrf.h"
#include "app_error.h"
#include "app_util_platform.h"
#include "app_error.h"
#include <string.h>
#include "port_platform.h"
#include "deca_types.h"
#include "deca_param_types.h"
#include "deca_regs.h"
#include "deca_device_api.h"
#include "UART.h"
#include "shared_var.h"
#include "app_uart.h"

    
/* DW1000 configuration */
static dwt_config_t config = {
    5,               /* Channel */
    DWT_PRF_64M,     /* PRF */
    DWT_PLEN_128,    /* Preamble length (TX) */
    DWT_PAC8,        /* PAC size (RX) */
    10,              /* TX preamble code (TX) */
    10,              /* RX preamble code (RX) */
    0,               /* Use standard SFD */
    DWT_BR_6M8,      /* Data rate */
    DWT_PHRMODE_STD, /* PHY header mode */
    (129 + 8 - 8)    /* SFD timeout (RX) */
};

/* Preamble timeout (PAC size multiples) */
#define PRE_TIMEOUT 1000

/* Delay between frames (UWB us) */
#define POLL_TX_TO_RESP_RX_DLY_UUS 100 

/* Antenna delays - CALIBRATE THESE */
#define TX_ANT_DLY 16300
#define RX_ANT_DLY 16456     

#define TASK_DELAY        200         /**< Task delay (ms) */
#define TIMER_PERIOD      2000        /**< Timer period (ms) */

#ifdef USE_FREERTOS

TaskHandle_t  ss_initiator_task_handle;
extern void ss_initiator_task_function (void * pvParameter);
TaskHandle_t  led_toggle_task_handle;
TimerHandle_t led_toggle_timer_handle;
#endif

#ifdef USE_FREERTOS

/* LED0 task entry function */
static void led_toggle_task_function (void * pvParameter)
{
  UNUSED_PARAMETER(pvParameter);
  while (true)
  {
    LEDS_INVERT(BSP_LED_0_MASK);
    vTaskDelay(TASK_DELAY);
    /* Tasks must be implemented to never return... */
  }
}

/* LED1 timer callback */
static void led_toggle_timer_callback (void * pvParameter)
{
  UNUSED_PARAMETER(pvParameter);
  LEDS_INVERT(BSP_LED_1_MASK);
}
#else

  extern int ss_init_run(void);
  extern int ss_resp_run(void);

#endif  // #ifdef USE_FREERTOS

// UART command buffer
char c[2] ; 
// Start ('S') and Finish ('F') commands
char aux[3] = {'S','F','\0'} ;  

int main(void)
{
  /* Setup debug LEDs */
  LEDS_CONFIGURE(BSP_LED_0_MASK | BSP_LED_1_MASK | BSP_LED_2_MASK);
  LEDS_ON(BSP_LED_0_MASK | BSP_LED_1_MASK | BSP_LED_2_MASK );

  #ifdef USE_FREERTOS
    /* Create LED0 blink task */
    UNUSED_VARIABLE(xTaskCreate(led_toggle_task_function, "LED0", configMINIMAL_STACK_SIZE + 200, NULL, 2, &led_toggle_task_handle));

    /* Start timer for LED1 blinking */
    led_toggle_timer_handle = xTimerCreate( "LED1", TIMER_PERIOD, pdTRUE, NULL, led_toggle_timer_callback);
    UNUSED_VARIABLE(xTimerStart(led_toggle_timer_handle, 0));

    /* Create SS TWR Initiator task */
    UNUSED_VARIABLE(xTaskCreate(ss_initiator_task_function, "SSTWR_INIT", configMINIMAL_STACK_SIZE + 200, NULL, 2, &ss_initiator_task_handle));
  #endif // #ifdef USE_FREERTOS
  
  /* Setup DW1000 IRQ pin */  
  nrf_gpio_cfg_input(DW1000_IRQ, NRF_GPIO_PIN_NOPULL);
  
  /* Init UART */
  boUART_Init ();
  printf("Singled Sided Two Way Ranging INIT\r\n");
  
  /* Reset DW1000 */
  reset_DW1000(); 

  /* Set SPI clock to 2MHz */
  port_set_dw1000_slowrate();        
  
  /* Init the DW1000 */
  if (dwt_initialise(DWT_LOADUCODE) == DWT_ERROR)
  {
    // DW1000 Init Failed
    while (1) {};
  }

  // Set SPI to 8MHz clock
  port_set_dw1000_fastrate();

  /* Configure DW1000. */
  dwt_configure(&config);

  /* Apply antenna delay values. */
  dwt_setrxantennadelay(RX_ANT_DLY);
  dwt_settxantennadelay(TX_ANT_DLY);
  
  /* Set RX delay and timeout. */
  dwt_setrxaftertxdelay(POLL_TX_TO_RESP_RX_DLY_UUS);
  dwt_setrxtimeout(65000); // Maximum value timeout with DW1000 is 65ms  
  
  #ifdef USE_FREERTOS     
    /* Start FreeRTOS scheduler. */
    vTaskStartScheduler();   

    while(1) 
    {};
  #else

    // No RTOS, run simple loop.
    while (1)
    {
      boUART_getc(c);
            
      if(c[0] == aux[0]) //start command received
      { 
          ss_init_run();
      }

      if(c[0] == aux[1]) //finish command received
      {
        printf("END");
        printf("\r\n");
        break;
      }
        
    } //while

  #endif
} //main