#ifndef YUV422_CONVERTER_VULKAN_H
#define YUV422_CONVERTER_VULKAN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Initialize Vulkan compute pipeline
 * Returns 0 on success, -1 on failure
 */
int yuv422_vulkan_init(uint32_t width, uint32_t height);

/* Convert 40-bit packed YUV422 to P010 using Vulkan compute
 * 
 * Input: dmabuf fd containing 40-bit packed YUV422 (9600x2160 bytes for 4K)
 * Output: dmabuf fd for P010 format (Y plane + UV plane)
 * 
 * This is ZERO-COPY - GPU writes directly to output dmabuf
 * 
 * Returns 0 on success, -1 on failure
 */
/* Convert with TRUE zero-copy - output dmabuf remains valid after return
 * 
 * CRITICAL: This function imports the output dmabuf into Vulkan and keeps
 * the VkDeviceMemory alive. You MUST call yuv422_vulkan_release_output() 
 * when the encoder is done with the dmabuf, otherwise you'll leak GPU memory
 * and eventually run out of resources.
 * 
 * Input: dmabuf fd containing 40-bit packed YUV422
 * Output: dmabuf fd for P010 format (Y plane + UV plane) - REMAINS VALID
 * 
 * Returns 0 on success, -1 on failure
 */
int yuv422_vulkan_convert_dmabuf(int in_fd, int out_fd, uint32_t width, uint32_t height);

/* Non-blocking submit — dispatches GPU work, returns immediately.
 * Must call yuv422_vulkan_convert_wait() before the next submit.
 * Returns 0 on success, -1 on failure.
 */
int yuv422_vulkan_convert_submit(int in_fd, int out_fd, uint32_t width, uint32_t height);

/* Wait for previously submitted GPU work to complete.
 * Returns 0 on success (or if nothing pending), -1 on failure.
 */
int yuv422_vulkan_convert_wait(void);

/* Release output resources after encoder is done
 * 
 * Call this after the encoder has finished reading the output dmabuf.
 * This frees the VkDeviceMemory associated with the output.
 * 
 * Returns 0 on success, -1 if no output to release
 */
int yuv422_vulkan_release_output(void);

/* Convert using triple-buffered async pipeline
 * 
 * This version overlaps GPU computation with encoder operations
 * for maximum throughput (target: 4K60)
 * 
 * slot: 0, 1, or 2 for triple buffering
 * fence_fd: output fence fd that signals when GPU is done (can wait on with poll())
 * 
 * Returns 0 on success, -1 on failure
 */
int yuv422_vulkan_convert_async(int in_fd, int out_fd, uint32_t width, uint32_t height,
                                 int slot, int *fence_fd);

/* Wait for async conversion to complete
 * 
 * timeout_ms: maximum time to wait in milliseconds
 * Returns 0 if complete, -1 if timeout
 */
int yuv422_vulkan_wait_async(int slot, int timeout_ms);

/* Cleanup Vulkan resources */
void yuv422_vulkan_cleanup(void);

/* Get last error message */
const char *yuv422_vulkan_last_error(void);

#ifdef __cplusplus
}
#endif

#endif /* YUV422_CONVERTER_VULKAN_H */
