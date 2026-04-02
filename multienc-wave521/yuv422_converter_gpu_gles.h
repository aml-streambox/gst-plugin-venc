#ifndef _YUV422_CONVERTER_GPU_GLES_H_
#define _YUV422_CONVERTER_GPU_GLES_H_

#include <stdint.h>
#include "yuv422_converter_neon.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GpuCtx GpuCtx;

/* Initialize GLES compute pipeline
 * Returns context pointer on success, NULL on failure
 * The caller owns the context and must call yuv422_gpu_gles_cleanup(ctx) to free it
 */
GpuCtx *yuv422_gpu_gles_init(void);

void yuv422_gpu_gles_cleanup(GpuCtx *ctx);
int yuv422_gpu_gles_convert_p010(GpuCtx *ctx, const ConversionParams *params);
int yuv422_gpu_gles_convert_p010_dmabuf(GpuCtx *ctx, int in_fd, int out_fd, uint32_t width, uint32_t height);
int yuv422_gpu_gles_compute_p010_dmabuf(GpuCtx *ctx, int in_fd, uint32_t width, uint32_t height);
const char *yuv422_gpu_gles_last_error(GpuCtx *ctx);

/* Repack P010 to Wave521 MSB format (MSB-aligned, byte-swapped) */
void repack_p010_to_wave5_msb(uint16_t *y, uint16_t *uv, uint32_t width, uint32_t height);

#ifdef __cplusplus
}
#endif

#endif
