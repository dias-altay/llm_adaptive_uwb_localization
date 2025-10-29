/*! ----------------------------------------------------------------------------
 * @file    ss_init_main.c
 * @brief   Single-sided two-way ranging (SS TWR) initiator.
 *
 * This code is for TAG LOCALIZATION, based on Decawave's SS TWR example.
 * It acts as an initiator (tag) in the network.
 * It ranges with every anchor, calculates and stores the distances
 * and standard deviations for each, and then publishes the estimations.
 *
 * @attention
 * Copyright 2015 (c) Decawave Ltd, Dublin, Ireland. All rights reserved.
 * @author Decawave
 * --------------------------------------------------------------------------- */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "FreeRTOS.h"
#include "task.h"
#include "deca_device_api.h"
#include "deca_regs.h"
#include "port_platform.h"
#include "app_uart.h"
#include "shared_var.h"

// Tag ID (0=tag, 1-N=responders). SET THIS BEFORE FLASHING.
int my_id = 0; 
#define  MAX_ITER 3      // Measurements per responder
#define NUM_ANCHORS 6    // Number of responders in network

// Explicit anchor ID rotation order (adjust to your responders)
static const uint8 ANCH_ID[NUM_ANCHORS] = { 1, 2, 3, 4, 5, 6 };
static int a_idx = 0;        // current anchor index [0..NUM_ANCHORS-1]

// Sequence number for MEAS/MISS logs
static uint32_t g_seq = 0;

#define APP_NAME "SS TWR INIT v1.3"

/* Ranging delay (ms) */
  #define RNG_DELAY_MS 1
  #define LAST_DELAY_MS 30
/* Frame format: 5,6=dest, 7,8=source */
/* Ranging frames (Initiator) */
static uint8 tx_poll_msg[] = {0x41, 0x88, 0, 0xCA, 0xDE, 'R', 'X', 'I', 'X', 'S',  0, 0, 0, 0, 0, 0, 0, 0};
static uint8 tx_poll_msg2[] = {0x41, 0x88, 0, 0xCA, 0xDE, 'R', 'X', 'I', 'X', 'F', 0, 0, 0, 0, 0, 0, 0, 0}; // Finish message

/* Ranging frames (Responder) */
static uint8 tx_resp_msg[] = {0x41, 0x88, 0, 0xCA, 0xDE, 'I', 'X', 'R', 'X', 0xE1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

/* Message common length */
#define ALL_MSG_COMMON_LEN 10
/* Frame field indexes */
#define ALL_MSG_SN_IDX 2
#define RESP_MSG_POLL_RX_TS_IDX 10
#define RESP_MSG_RESP_TX_TS_IDX 14
#define RESP_MSG_TS_LEN 4

#define SOURCE_IDX 8
#define DEST_IDX 6

#define MEAN_IDX 12
#define STD_IDX 14
/* Frame sequence number */
static uint8 frame_seq_nb = 0;

/* RX response buffer */
#define RX_BUF_LEN 25
static uint8 rx_buffer[RX_BUF_LEN];
#define RX_BUF_LEN2 29
static uint8 rx_buffer2[RX_BUF_LEN2];

// TX timeout extended for nRF operation
#define POLL_RX_TO_RESP_TX_DLY_UUS  1100

/* Delay from TX end to RX enable (DWT_RESPONSE_EXPECTED) */
#define RESP_TX_TO_FINAL_RX_DLY_UUS 500

/* 40-bit timestamps need 64-bit vars */
typedef signed long long int64;
typedef unsigned long long uint64;
static uint64 poll_rx_ts;

static uint64 get_rx_timestamp_u64(void);
static void resp_msg_set_ts(uint8 *ts_field, const uint64 ts);

/* 40-bit timestamps need 64-bit vars */
typedef unsigned long long uint64;
static uint64 poll_rx_ts;
static uint64 resp_tx_ts;

/* Status register copy for debugging */
static uint32 status_reg = 0;

/* UUS to DWT time conversion */
#define UUS_TO_DWT_TIME 65536

/* Speed of light (m/s) */
#define SPEED_OF_LIGHT 299702547

/* Computed ToF and distance for debugging */
static double tof;
static double distance;

static void resp_msg_get_ts(uint8 *ts_field, uint32 *ts);


/* Transaction counters */
static volatile int tx_count = 0 ; // Successful transmit counter

/*Iteration counter*/
int iter_cnt = 0;
/* Responder timeout counter */
int timeout_resp = 0;

/* Measurements matrix: [anchor][data] */
/* Each row: [tag_id, anchor_id, mean(2), std(2), pos(3), err(1)] */
uint8 measurements_matrix[NUM_ANCHORS][10] = {0}; 

/*Measurements matrix access indexes*/
#define M1_IDX 2  //Mean index
#define S1_IDX 4  //Standard deviation index
#define POS_IDX 6 //Anchor position index

// Mean/Std calculation vars
float mean = 0;
uint8 mean_lsb = 0;
uint8 mean_msb = 0;
float total = 0;
float SD = 0;
uint8 SD_lsb = 0;
uint8 SD_msb = 0;

//Measurements vector
float measurements[MAX_ITER];
// Current responder ID
int resp_id = 1;

/* --- UART Logging Helpers --- */
static void uart_write(const char* s, int n) {
  for (int i = 0; i < n; i++) {
    while (app_uart_put((uint8_t)s[i]) != NRF_SUCCESS) {
      /* Busy: wait until there is space in the TX FIFO */
      /* Optionally: nrf_delay_us(50); */
    }
  }
}
static void log_meas_csv_uart(uint32_t seq, uint8 anchor_id, float dist_m,
                              uint32 poll_tx_ts, uint32 resp_rx_ts,
                              uint32 poll_rx_ts, uint32 resp_tx_ts,
                              float clk_off_ratio) {
  dwt_rxdiag_t diag; dwt_readdiagnostics(&diag);
  char buf[256];
  int n = snprintf(buf, sizeof(buf),
    "MEAS,%lu,%u,%.3f,%.6f,%lu,%lu,%lu,%lu,%u,%u,%u,%u,%u,%u\r\n",
    (unsigned long)seq, (unsigned)anchor_id, dist_m, clk_off_ratio,
    (unsigned long)poll_tx_ts, (unsigned long)resp_rx_ts,
    (unsigned long)poll_rx_ts, (unsigned long)resp_tx_ts,
    (unsigned)diag.firstPathAmp1, (unsigned)diag.firstPathAmp2, (unsigned)diag.firstPathAmp3,
    (unsigned)diag.maxNoise, (unsigned)diag.stdNoise, (unsigned)diag.rxPreamCount);
  uart_write(buf, n);
}
// reason: 1=timeout, 2=rx_error
static void log_miss_csv_uart(uint32_t seq, uint8 anchor_id, int reason) {
  char buf[64];
  int n = snprintf(buf, sizeof(buf), "MISS,%lu,%u,%d\r\n",
                   (unsigned long)seq, (unsigned)anchor_id, reason);
  uart_write(buf, n);
}

/*! ------------------------------------------------------------------------------------------------------------------
 * @fn ss_init_run()
 *
 * @brief SS TWR Initiator main loop function.
 *
 * ------------------------------------------------------------------------------------------------------------------ */
int ss_init_run(void)
{
  tx_poll_msg[SOURCE_IDX] = my_id;
  tx_poll_msg2[SOURCE_IDX] = my_id;

  if((iter_cnt >= MAX_ITER) || (timeout_resp == MAX_ITER * 5))
  {
    measurements_matrix[resp_id-1][0] = my_id;   //tag id
    measurements_matrix[resp_id-1][1] = resp_id; //anchor id

    if(timeout_resp == MAX_ITER * 5) //Timeout: fill with 255
    {
      measurements_matrix[resp_id - 1][M1_IDX]= 255;
      measurements_matrix[resp_id - 1][M1_IDX + 1]= 255;
      measurements_matrix[resp_id - 1][S1_IDX]= 255;
      measurements_matrix[resp_id - 1][S1_IDX + 1]= 255;
    }

    else // OK: calculate and store mean/std
    {
      mean = total/MAX_ITER;
      mean_lsb = (int)(mean*100) % 256;
      mean_msb = (int)(mean*100 / 256);
      total = 0;
    
      SD = 0;
      for (int i = 0; i < MAX_ITER; ++i)
      {
          SD += pow(measurements[i] - mean, 2);
      }
      SD = sqrt(SD / MAX_ITER);
    
      SD_lsb = (int)(SD*100) % 256;
      SD_msb = (int)(SD*100 / 256);
      
      measurements_matrix[resp_id - 1][M1_IDX]= mean_msb;
      measurements_matrix[resp_id - 1][M1_IDX + 1]= mean_lsb;
      measurements_matrix[resp_id - 1][S1_IDX]= SD_msb;
      measurements_matrix[resp_id - 1][S1_IDX + 1]= SD_lsb;
    
    }
    
    measurements_matrix[resp_id-1][POS_IDX] = rx_buffer[18]; //POS X 
    measurements_matrix[resp_id-1][POS_IDX+1] = rx_buffer[19];//POS Y
    measurements_matrix[resp_id-1][POS_IDX+2] = rx_buffer[20];//POS Z
    measurements_matrix[resp_id-1][POS_IDX+3] = rx_buffer[21];//ERROR

    a_idx = (a_idx + 1) % NUM_ANCHORS;  // minimal change: rotate index instead

    //reset iteration counter and timeout
    iter_cnt = 0;
    timeout_resp = 0;

    if (a_idx == 0) // All responders asked: print matrix and reset
    {
      for(int i = 0; i < NUM_ANCHORS; i++)
      {
        for(int j = 0; j < 10; j++)
        {
            if((i == NUM_ANCHORS-1) && (j == 9))
              printf("%d", measurements_matrix[i][j]);
              
            else
              printf("%d,", measurements_matrix[i][j]);
            
            measurements_matrix[i][j] = 0;
        }
       }
       printf("\r\n");
    }
  }
  
  else
  {
      /*Set poll msg destination to the target responder*/
    resp_id = ANCH_ID[a_idx];           // minimal change: choose current anchor ID
    tx_poll_msg[DEST_IDX] = resp_id;
    /* Write frame data and TX config. */
    tx_poll_msg[ALL_MSG_SN_IDX] = frame_seq_nb;
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_TXFRS);
    dwt_writetxdata(sizeof(tx_poll_msg), tx_poll_msg, 0); /* Zero offset in TX buffer. */
    dwt_writetxfctrl(sizeof(tx_poll_msg), 0, 1); /* Zero offset in TX buffer, ranging. */
  }

  /* Start TX, response expected. */
  dwt_starttx(DWT_START_TX_IMMEDIATE | DWT_RESPONSE_EXPECTED);
  tx_count++;
  
  /* Poll for RX frame, error, or timeout. */
  while (!((status_reg = dwt_read32bitreg(SYS_STATUS_ID)) & (SYS_STATUS_RXFCG | SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR)))
  {};

    #if 0  // include if required to help debug timeouts.
    int temp = 0;     
    if(status_reg & SYS_STATUS_RXFCG )
    temp =1;
    else if(status_reg & SYS_STATUS_ALL_RX_TO )
    temp =2;
    if(status_reg & SYS_STATUS_ALL_RX_ERR )
    temp =3;
    #endif

  /* Increment frame sequence number (modulo 256). */
  frame_seq_nb++;
  
  /* Increment responder timeout (fail-safe). */
  timeout_resp++;

  if (status_reg & SYS_STATUS_RXFCG)
  {    
    uint32 frame_len;  
    
    /* Clear good RX frame event in the DW1000 status register. */
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_RXFCG);

    /* A frame has been received, read it into the local buffer. */
    frame_len = dwt_read32bitreg(RX_FINFO_ID) & RX_FINFO_RXFLEN_MASK;
    
    if (frame_len <= RX_BUF_LEN)
    {
      dwt_readrxdata(rx_buffer, frame_len, 0);
    }

    /* Check if frame is the expected response. */
    rx_buffer[ALL_MSG_SN_IDX] = 0;
    
    /* Check source and destination IDs. */
    if((rx_buffer[SOURCE_IDX]==resp_id) && (rx_buffer[DEST_IDX] == my_id) && (rx_buffer[DEST_IDX - 1] == 'I'))
    {
      /* Valid message, reset timeout. */
      timeout_resp = 0;
      
      uint32 poll_tx_ts, resp_rx_ts, poll_rx_ts, resp_tx_ts;
      int32 rtd_init, rtd_resp;
      float clockOffsetRatio ;

      /* Retrieve TX/RX timestamps. */
      poll_tx_ts = dwt_readtxtimestamplo32();
      resp_rx_ts = dwt_readrxtimestamplo32();

      /* Calculate clock offset ratio. */
      clockOffsetRatio = dwt_readcarrierintegrator() * (FREQ_OFFSET_MULTIPLIER * HERTZ_TO_PPM_MULTIPLIER_CHAN_5 / 1.0e6) ;

      /* Get timestamps embedded in response message. */
      resp_msg_get_ts(&rx_buffer[RESP_MSG_POLL_RX_TS_IDX], &poll_rx_ts);
      resp_msg_get_ts(&rx_buffer[RESP_MSG_RESP_TX_TS_IDX], &resp_tx_ts);

      /* Compute ToF and distance, correcting for clock offset */
      rtd_init = resp_rx_ts - poll_tx_ts;
      rtd_resp = resp_tx_ts - poll_rx_ts;

      tof = ((rtd_init - rtd_resp * (1.0f - clockOffsetRatio)) / 2.0f) * DWT_TIME_UNITS; // Specifying 1.0f and 2.0f are floats to clear warning 
      distance = tof * SPEED_OF_LIGHT;

      /* per-measurement UART log */
      log_meas_csv_uart(g_seq++, (uint8)resp_id, (float)distance,
                         poll_tx_ts, resp_rx_ts, poll_rx_ts, resp_tx_ts,
                         clockOffsetRatio);
      
      /* Store measurement for avg/std calc. */
      measurements[iter_cnt] = distance;
      total += measurements[iter_cnt];
      iter_cnt++;
    }
  }
  else
  {
    /* Clear RX error/timeout events in the DW1000 status register. */
    dwt_write32bitreg(SYS_STATUS_ID, SYS_STATUS_ALL_RX_TO | SYS_STATUS_ALL_RX_ERR);

    /* Reset RX to properly reinitialise LDE operation. */
    dwt_rxreset();

    /* per-miss UART log (1=timeout, 2=rx_error) */
    log_miss_csv_uart(g_seq, (uint8)resp_id, (status_reg & SYS_STATUS_ALL_RX_TO) ? 1 : 2);
  }

  /* Execute a delay between ranging exchanges. */
  deca_sleep(RNG_DELAY_MS);
}

/*! ------------------------------------------------------------------------------------------------------------------
 * @fn resp_msg_get_ts()
 *
 * @brief Read a 32-bit timestamp from a response message buffer.
 *
 * ------------------------------------------------------------------------------------------------------------------ */
static void resp_msg_get_ts(uint8 *ts_field, uint32 *ts)
{
  int i;
  *ts = 0;
  for (i = 0; i < RESP_MSG_TS_LEN; i++)
  {
    *ts += ts_field[i] << (i * 8);
  }
}


/*! ------------------------------------------------------------------------------------------------------------------
 * @fn get_rx_timestamp_u64()
 *
 * @brief Get the 40-bit RX timestamp as a 64-bit variable.
 *
 * ------------------------------------------------------------------------------------------------------------------ */
static uint64 get_rx_timestamp_u64(void)
{
  uint8 ts_tab[5];
  uint64 ts = 0;
  int i;
  dwt_readrxtimestamp(ts_tab);
  for (i = 4; i >= 0; i--)
  {
    ts <<= 8;
    ts |= ts_tab[i];
  }
  return ts;
}

/*! ------------------------------------------------------------------------------------------------------------------
 * @fn resp_msg_set_ts()
 *
 * @brief Write a 64-bit timestamp into a 32-bit message buffer field.
 *
 * ------------------------------------------------------------------------------------------------------------------ */
static void resp_msg_set_ts(uint8 *ts_field, const uint64 ts)
{
  int i;
  for (i = 0; i < RESP_MSG_TS_LEN; i++)
  {
    ts_field[i] = (ts >> (i * 8)) & 0xFF;
  }
}