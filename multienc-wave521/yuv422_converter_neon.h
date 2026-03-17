/*
 * YUV422 10-bit Converter API Header
 * 
 * Provides conversion from VDIN 40-bit packed format to:
 * - P010 (YUV420 10-bit)
 * - Wave521 32-bit packed 420 10-bit format
 */

#ifndef _YUV422_CONVERTER_H_
#define _YUV422_CONVERTER_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Output format types */
typedef enum {
    OUTPUT_YUV422_16BIT = 0,
    OUTPUT_YUV420_P010 = 1,
    OUTPUT_YUV420_WAVE521_32BIT = 2  /* Wave521 32-bit packed 420 10-bit */
} OutputFormat;

/* CPU core selection */
typedef enum {
    CORE_SMALL_ONLY = 0,   /* Use only small cores (0-5) */
    CORE_BIG_ONLY = 1,     /* Use only big cores (6-7) */
    CORE_BOTH = 2          /* Use both small and big cores */
} CoreSelection;

/* Thread configuration */
#define MAX_THREADS_SMALL 6
#define MAX_THREADS_BIG 2
#define MAX_THREADS_BOTH 8
#define DEFAULT_THREADS 6

/* Conversion parameters */
typedef struct {
    uint32_t width;
    uint32_t height;
    const uint8_t *input_data;   /* Input: 40-bit packed YUV422 (uncached) */
    uint8_t *staging_data;       /* Staging buffer for cached input copy (NULL = direct) */
    uint16_t *output_y;          /* Output Y plane (16-bit per pixel for P010) */
    uint16_t *output_uv;         /* Output UV plane (16-bit per component for P010) */
    uint32_t *output_y_wave521;  /* Output Y plane (32-bit packed for wave521) */
    uint32_t *output_uv_wave521; /* Output UV plane (32-bit packed for wave521) */
    OutputFormat output_format;
    /* Thread configuration */
    uint32_t num_threads;        /* Number of threads to use */
    CoreSelection core_selection;/* Which cores to use */
} ConversionParams;

/* Multi-threaded NEON implementation (configurable cores) */
int yuv422_10bit_convert_neon(const ConversionParams *params);

/* Persistent worker pool controls for P010 path */
int yuv422_10bit_thread_pool_configure(uint32_t num_threads, CoreSelection core_selection);
void yuv422_10bit_thread_pool_destroy(void);

/* Wave521 32-bit packed 420 10-bit format - NEON optimized */
int yuv422_10bit_convert_wave521(const ConversionParams *params);

/* Scalar CPU implementation (fallback) */
int yuv422_10bit_convert_cpu(const ConversionParams *params);
int yuv422_10bit_convert_wave521_cpu(const ConversionParams *params);



#ifdef __cplusplus
}
#endif

#endif /* _YUV422_CONVERTER_H_ */
