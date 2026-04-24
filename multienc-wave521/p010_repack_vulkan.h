#ifndef P010_REPACK_VULKAN_H
#define P010_REPACK_VULKAN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct P010RepackCtx P010RepackCtx;

P010RepackCtx *p010_repack_vulkan_init(void);
int p010_repack_vulkan_convert_inplace(P010RepackCtx *ctx, int fd, uint32_t width, uint32_t height);
void p010_repack_vulkan_cleanup(P010RepackCtx *ctx);
const char *p010_repack_vulkan_last_error(P010RepackCtx *ctx);

#ifdef __cplusplus
}
#endif

#endif
