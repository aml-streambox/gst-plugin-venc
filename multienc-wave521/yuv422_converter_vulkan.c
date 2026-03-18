/*
 * Vulkan-based Zero-Copy YUV422 to P010 Converter
 * 
 * Optimized for Mali-G52 (Bifrost) on Amlogic T7/T7C:
 * - Cached dmabuf imports (both input and output — no per-frame alloc/free)
 * - DMA_BUF_IOCTL_SYNC on input to invalidate GPU cache for fresh frame data
 * - uint32 reads + packed uint32 writes in shader
 * - 2D dispatch eliminates integer divides
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <vulkan/vulkan.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>
#include <errno.h>
#include <stdbool.h>
#include <linux/dma-buf.h>
#include <sys/ioctl.h>

#include "yuv422_converter_vulkan.h"

/* =============================================================================
 * Configuration Constants
 * ============================================================================= */

/* Set to 1 to enable verbose per-frame logging (fstat, buffer sizes, etc.) */
#ifndef VULKAN_DEBUG
#define VULKAN_DEBUG 0
#endif

#define MAX_FRAMES_IN_FLIGHT 3
#define DMABUF_CACHE_SIZE 8   /* Max cached dmabuf imports (input pool) */

#define VK_CHECK(result, msg) do { \
    if (result != VK_SUCCESS) { \
        snprintf(ctx.last_error, sizeof(ctx.last_error), "%s: %d", msg, result); \
        fprintf(stderr, "[VULKAN-ERR] %s\n", ctx.last_error); \
        return -1; \
    } \
} while(0)

/* =============================================================================
 * Shader Code (unused — we load pre-compiled SPIR-V)
 * ============================================================================= */

static const char *compute_shader_glsl = "(see yuv422_to_p010.comp)";

/* =============================================================================
 * DMA-buf Import Cache
 * ============================================================================= */

typedef struct {
    int fd;                /* original dmabuf fd (NOT dup'd — we don't own it) */
    int fd_dup;            /* dup'd fd we passed to Vulkan (Vulkan owns it) */
    VkBuffer buffer;
    VkDeviceMemory memory;
    VkDeviceSize size;
    int valid;
    uint64_t last_used;    /* frame counter for LRU eviction */
} DmabufCacheEntry;

/* =============================================================================
 * Vulkan Context Structure
 * ============================================================================= */

typedef struct {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    uint32_t compute_queue_family;
    VkCommandPool command_pool;
    VkDescriptorPool descriptor_pool;
    VkPipeline pipeline;
    VkPipelineLayout pipeline_layout;
    VkDescriptorSetLayout descriptor_set_layout;
    VkShaderModule shader_module;
    
    /* Triple buffering resources */
    VkCommandBuffer command_buffers[MAX_FRAMES_IN_FLIGHT];
    VkDescriptorSet descriptor_sets[MAX_FRAMES_IN_FLIGHT];
    VkFence fences[MAX_FRAMES_IN_FLIGHT];
    VkSemaphore semaphores[MAX_FRAMES_IN_FLIGHT];
    int slot_busy[MAX_FRAMES_IN_FLIGHT];
    
    /* Device properties */
    VkPhysicalDeviceMemoryProperties memory_props;
    
    /* Configuration */
    uint32_t width;
    uint32_t height;
    int initialized;
    uint64_t frame_count;
    char last_error[256];
    
    /* Cached output dmabuf import (same fd every frame) */
    DmabufCacheEntry cached_output;
    
    /* Cached input dmabuf imports (fd-keyed pool with DMA_BUF_SYNC) */
    DmabufCacheEntry input_cache[DMABUF_CACHE_SIZE];
    int input_cache_count;
    
    /* Track which input cache entry is active this frame */
    int active_input_idx;  /* index into input_cache[], or -1 */
    
    /* Pending submit state for split submit/wait API */
    int pending_in_fd;     /* input fd needing DMA_BUF_SYNC_END after wait, or -1 */
    int has_pending;       /* 1 if a submit is in-flight awaiting wait */
} VulkanCtx;

static VulkanCtx ctx = {0};

/* =============================================================================
 * Helper Functions
 * ============================================================================= */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

static int find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags props) {
    for (uint32_t i = 0; i < ctx.memory_props.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && 
            (ctx.memory_props.memoryTypes[i].propertyFlags & props) == props) {
            return i;
        }
    }
    return -1;
}

/* Embed pre-compiled SPIR-V shader */
#include "yuv422_to_p010_spv.h"

static int load_shader(VkShaderModule *shader_module) {
    VkShaderModuleCreateInfo create_info = {
        .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = sizeof(yuv422_to_p010_spv),
        .pCode = (const uint32_t *)yuv422_to_p010_spv,
    };
    
    VkResult result = vkCreateShaderModule(ctx.device, &create_info, NULL, shader_module);
    if (result != VK_SUCCESS) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), 
                 "vkCreateShaderModule failed: %d", result);
        return -1;
    }
    
    fprintf(stderr, "[VULKAN] Loaded shader: %zu bytes\n", sizeof(yuv422_to_p010_spv));
    return 0;
}

/* =============================================================================
 * DMA-buf Import (raw — no caching)
 * ============================================================================= */

/* Import dmabuf as Vulkan buffer.
 * IMPORTANT: This function TAKES OWNERSHIP of 'fd' — Vulkan will close it.
 * Caller must pass a dup()'d fd if they need to keep the original.
 */
static int import_dmabuf(int fd, VkDeviceSize size, VkBuffer *buffer, VkDeviceMemory *memory) {
    VkExternalMemoryBufferCreateInfo ext_mem_info = {
        .sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
        .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
    };
    
    VkBufferCreateInfo buffer_info = {
        .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .pNext = &ext_mem_info,
        .size = size,
        .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };
    
    VkResult result = vkCreateBuffer(ctx.device, &buffer_info, NULL, buffer);
    if (result != VK_SUCCESS) {
        close(fd);
        snprintf(ctx.last_error, sizeof(ctx.last_error), "vkCreateBuffer failed: %d", result);
        return -1;
    }
    
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(ctx.device, *buffer, &mem_reqs);
    
    VkImportMemoryFdInfoKHR import_info = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        .fd = fd,
    };
    
    VkDeviceSize alloc_size = mem_reqs.size;
    struct stat fd_stat;
    if (fstat(fd, &fd_stat) == 0 && (VkDeviceSize)fd_stat.st_size > alloc_size) {
        alloc_size = fd_stat.st_size;
    }
    
    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &import_info,
        .allocationSize = alloc_size,
        .memoryTypeIndex = find_memory_type(mem_reqs.memoryTypeBits, 0),
    };
    
    if (alloc_info.memoryTypeIndex == (uint32_t)-1) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "No suitable memory type");
        vkDestroyBuffer(ctx.device, *buffer, NULL);
        return -1;
    }
    
    result = vkAllocateMemory(ctx.device, &alloc_info, NULL, memory);
    if (result != VK_SUCCESS) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "vkAllocateMemory failed: %d", result);
        vkDestroyBuffer(ctx.device, *buffer, NULL);
        return -1;
    }
    
    result = vkBindBufferMemory(ctx.device, *buffer, *memory, 0);
    if (result != VK_SUCCESS) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "vkBindBufferMemory failed: %d", result);
        vkFreeMemory(ctx.device, *memory, NULL);
        vkDestroyBuffer(ctx.device, *buffer, NULL);
        return -1;
    }
    
    return 0;
}

/* =============================================================================
 * Cached DMA-buf Import/Lookup
 * ============================================================================= */

/* Destroy a single cache entry's Vulkan resources */
static void cache_entry_destroy(DmabufCacheEntry *entry) {
    if (!entry->valid) return;
    vkDestroyBuffer(ctx.device, entry->buffer, NULL);
    vkFreeMemory(ctx.device, entry->memory, NULL);
    entry->valid = 0;
    entry->fd = -1;
    entry->fd_dup = -1;
}

/* Look up or import an input dmabuf. Returns cache index, or -1 on failure. */
static int input_cache_get(int fd, VkDeviceSize size) {
    /* Check if we already have this fd cached */
    for (int i = 0; i < ctx.input_cache_count; i++) {
        if (ctx.input_cache[i].valid && ctx.input_cache[i].fd == fd) {
            ctx.input_cache[i].last_used = ctx.frame_count;
#if VULKAN_DEBUG
            fprintf(stderr, "[VULKAN] Input cache HIT fd=%d slot=%d\n", fd, i);
#endif
            return i;
        }
    }
    
    /* Cache miss — find a free slot or evict LRU */
    int slot = -1;
    if (ctx.input_cache_count < DMABUF_CACHE_SIZE) {
        slot = ctx.input_cache_count++;
    } else {
        /* Evict least recently used */
        uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < DMABUF_CACHE_SIZE; i++) {
            if (ctx.input_cache[i].last_used < oldest) {
                oldest = ctx.input_cache[i].last_used;
                slot = i;
            }
        }
        /* Must wait for GPU idle before destroying — caller ensures this */
        cache_entry_destroy(&ctx.input_cache[slot]);
    }
    
    /* Import the new fd */
    int fd_dup = dup(fd);
    if (fd_dup < 0) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "Failed to dup input fd %d: %s", fd, strerror(errno));
        return -1;
    }
    
    VkBuffer buffer;
    VkDeviceMemory memory;
    if (import_dmabuf(fd_dup, size, &buffer, &memory) != 0) {
        return -1;
    }
    
    ctx.input_cache[slot].fd = fd;
    ctx.input_cache[slot].fd_dup = fd_dup;
    ctx.input_cache[slot].buffer = buffer;
    ctx.input_cache[slot].memory = memory;
    ctx.input_cache[slot].size = size;
    ctx.input_cache[slot].valid = 1;
    ctx.input_cache[slot].last_used = ctx.frame_count;
    
#if VULKAN_DEBUG
    fprintf(stderr, "[VULKAN] Input cache MISS fd=%d slot=%d (total cached=%d)\n", fd, slot, ctx.input_cache_count);
#endif
    
    return slot;
}

/* Look up or import the output dmabuf. Returns 0 on success. */
static int output_cache_get(int fd, VkDeviceSize size) {
    if (ctx.cached_output.valid && ctx.cached_output.fd == fd) {
#if VULKAN_DEBUG
        fprintf(stderr, "[VULKAN] Output cache HIT fd=%d\n", fd);
#endif
        return 0;
    }
    
    /* New fd or first frame — destroy old and import */
    if (ctx.cached_output.valid) {
        cache_entry_destroy(&ctx.cached_output);
    }
    
    int fd_dup = dup(fd);
    if (fd_dup < 0) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "Failed to dup output fd %d: %s", fd, strerror(errno));
        return -1;
    }
    
    VkBuffer buffer;
    VkDeviceMemory memory;
    if (import_dmabuf(fd_dup, size, &buffer, &memory) != 0) {
        return -1;
    }
    
    ctx.cached_output.fd = fd;
    ctx.cached_output.fd_dup = fd_dup;
    ctx.cached_output.buffer = buffer;
    ctx.cached_output.memory = memory;
    ctx.cached_output.size = size;
    ctx.cached_output.valid = 1;
    
#if VULKAN_DEBUG
    fprintf(stderr, "[VULKAN] Output cache MISS fd=%d (imported)\n", fd);
#endif
    return 0;
}

/* =============================================================================
 * Initialization
 * ============================================================================= */

int yuv422_vulkan_init(uint32_t width, uint32_t height) {
    if (ctx.initialized) {
        return 0;
    }
    
    ctx.width = width;
    ctx.height = height;
    ctx.active_input_idx = -1;
    ctx.input_cache_count = 0;
    ctx.cached_output.valid = 0;
    ctx.cached_output.fd = -1;
    
    for (int i = 0; i < DMABUF_CACHE_SIZE; i++) {
        ctx.input_cache[i].valid = 0;
        ctx.input_cache[i].fd = -1;
    }
    
    /* Check for Vulkan support */
    uint32_t instance_version = 0;
    VkResult result = vkEnumerateInstanceVersion(&instance_version);
    if (result != VK_SUCCESS) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "Vulkan not supported");
        return -1;
    }
    
    fprintf(stderr, "[VULKAN] Instance version: %d.%d.%d\n",
            VK_VERSION_MAJOR(instance_version),
            VK_VERSION_MINOR(instance_version),
            VK_VERSION_PATCH(instance_version));
    
    /* Create Vulkan instance */
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "YUV422Converter",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "VulkanCompute",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_2,
    };
    
    const char *extensions[] = {
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
    };
    
    VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = 2,
        .ppEnabledExtensionNames = extensions,
    };
    
    result = vkCreateInstance(&instance_info, NULL, &ctx.instance);
    VK_CHECK(result, "vkCreateInstance");
    
    /* Select physical device */
    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(ctx.instance, &device_count, NULL);
    if (device_count == 0) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "No Vulkan devices found");
        return -1;
    }
    
    VkPhysicalDevice *devices = malloc(device_count * sizeof(VkPhysicalDevice));
    vkEnumeratePhysicalDevices(ctx.instance, &device_count, devices);
    
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        
        uint32_t queue_family_count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queue_family_count, NULL);
        VkQueueFamilyProperties *queue_families = malloc(queue_family_count * sizeof(VkQueueFamilyProperties));
        vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queue_family_count, queue_families);
        
        for (uint32_t j = 0; j < queue_family_count; j++) {
            if (queue_families[j].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                ctx.physical_device = devices[i];
                ctx.compute_queue_family = j;
                fprintf(stderr, "[VULKAN] Selected device: %s\n", props.deviceName);
                break;
            }
        }
        
        free(queue_families);
        if (ctx.physical_device != VK_NULL_HANDLE) break;
    }
    
    free(devices);
    
    if (ctx.physical_device == VK_NULL_HANDLE) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "No device with compute support");
        return -1;
    }
    
    vkGetPhysicalDeviceMemoryProperties(ctx.physical_device, &ctx.memory_props);
    
    /* Create logical device */
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx.compute_queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };
    
    const char *device_extensions[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_KHR_SHADER_NON_SEMANTIC_INFO_EXTENSION_NAME,
    };
    
    VkDeviceCreateInfo device_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_info,
        .enabledExtensionCount = 5,
        .ppEnabledExtensionNames = device_extensions,
    };
    
    result = vkCreateDevice(ctx.physical_device, &device_info, NULL, &ctx.device);
    VK_CHECK(result, "vkCreateDevice");
    
    vkGetDeviceQueue(ctx.device, ctx.compute_queue_family, 0, &ctx.compute_queue);
    
    /* Create command pool */
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = ctx.compute_queue_family,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                 VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    };
    
    result = vkCreateCommandPool(ctx.device, &pool_info, NULL, &ctx.command_pool);
    VK_CHECK(result, "vkCreateCommandPool");
    
    /* Allocate command buffers */
    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx.command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
    };
    
    result = vkAllocateCommandBuffers(ctx.device, &cmd_alloc, ctx.command_buffers);
    VK_CHECK(result, "vkAllocateCommandBuffers");
    
    /* Create fences */
    VkFenceCreateInfo fence_info = {
        .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
        .flags = VK_FENCE_CREATE_SIGNALED_BIT,
    };
    
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        result = vkCreateFence(ctx.device, &fence_info, NULL, &ctx.fences[i]);
        VK_CHECK(result, "vkCreateFence");
        ctx.slot_busy[i] = 0;
    }
    
    /* Create semaphores */
    VkSemaphoreCreateInfo semaphore_info = {
        .sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO,
    };
    
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        result = vkCreateSemaphore(ctx.device, &semaphore_info, NULL, &ctx.semaphores[i]);
        VK_CHECK(result, "vkCreateSemaphore");
    }
    
    /* Create descriptor pool */
    VkDescriptorPoolSize pool_sizes[] = {
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_FRAMES_IN_FLIGHT * 3},
    };
    
    VkDescriptorPoolCreateInfo desc_pool_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .maxSets = MAX_FRAMES_IN_FLIGHT,
        .poolSizeCount = 1,
        .pPoolSizes = pool_sizes,
    };
    
    result = vkCreateDescriptorPool(ctx.device, &desc_pool_info, NULL, &ctx.descriptor_pool);
    VK_CHECK(result, "vkCreateDescriptorPool");
    
    /* Create descriptor set layout */
    VkDescriptorSetLayoutBinding bindings[] = {
        {
            .binding = 0,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
        {
            .binding = 2,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        },
    };
    
    VkDescriptorSetLayoutCreateInfo layout_info = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .bindingCount = 3,
        .pBindings = bindings,
    };
    
    result = vkCreateDescriptorSetLayout(ctx.device, &layout_info, NULL, &ctx.descriptor_set_layout);
    VK_CHECK(result, "vkCreateDescriptorSetLayout");
    
    /* Allocate descriptor sets */
    VkDescriptorSetAllocateInfo desc_alloc = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .descriptorPool = ctx.descriptor_pool,
        .descriptorSetCount = MAX_FRAMES_IN_FLIGHT,
        .pSetLayouts = (VkDescriptorSetLayout[]){ctx.descriptor_set_layout, ctx.descriptor_set_layout, ctx.descriptor_set_layout},
    };
    
    result = vkAllocateDescriptorSets(ctx.device, &desc_alloc, ctx.descriptor_sets);
    VK_CHECK(result, "vkAllocateDescriptorSets");
    
    /* Create pipeline layout — push constants now include pairs_per_row */
    VkPushConstantRange push_constant = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(uint32_t) * 3,  /* width, height, pairs_per_row */
    };
    
    VkPipelineLayoutCreateInfo pipeline_layout_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount = 1,
        .pSetLayouts = &ctx.descriptor_set_layout,
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant,
    };
    
    result = vkCreatePipelineLayout(ctx.device, &pipeline_layout_info, NULL, &ctx.pipeline_layout);
    VK_CHECK(result, "vkCreatePipelineLayout");
    
    /* Load shader */
    result = load_shader(&ctx.shader_module);
    if (result != 0) {
        return -1;
    }
    
    /* Create compute pipeline */
    VkComputePipelineCreateInfo pipeline_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = ctx.shader_module,
            .pName = "main",
        },
        .layout = ctx.pipeline_layout,
    };
    
    result = vkCreateComputePipelines(ctx.device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &ctx.pipeline);
    VK_CHECK(result, "vkCreateComputePipelines");
    
    ctx.initialized = 1;
    ctx.frame_count = 0;
    fprintf(stderr, "[VULKAN] Initialization complete\n");
    
    return 0;
}

/* =============================================================================
 * Non-blocking Submit — dispatches GPU work, returns immediately
 * ============================================================================= */

int yuv422_vulkan_convert_submit(int in_fd, int out_fd, uint32_t width, uint32_t height) {
    if (!ctx.initialized) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "Vulkan not initialized");
        return -1;
    }
    
    VkResult result;
    
    /* Fence management */
    if (ctx.frame_count == 0) {
        vkResetFences(ctx.device, 1, &ctx.fences[0]);
    }
    
    /* Calculate buffer sizes */
    VkDeviceSize input_size = (VkDeviceSize)width * height * 5 / 2;
    VkDeviceSize y_plane_size = (VkDeviceSize)width * height * 2;
    VkDeviceSize uv_plane_size = (VkDeviceSize)width * height;
    VkDeviceSize output_size = y_plane_size + uv_plane_size;
    
#if VULKAN_DEBUG
    struct stat in_stat, out_stat;
    if (fstat(in_fd, &in_stat) == 0) {
        fprintf(stderr, "[VULKAN] Input dmabuf fd=%d actual_size=%ld expected=%lu\n",
                in_fd, (long)in_stat.st_size, (unsigned long)input_size);
    }
    if (fstat(out_fd, &out_stat) == 0) {
        fprintf(stderr, "[VULKAN] Output dmabuf fd=%d actual_size=%ld expected=%lu\n",
                out_fd, (long)out_stat.st_size, (unsigned long)output_size);
    }
    fprintf(stderr, "[VULKAN] Buffer sizes: input=%lu output=%lu (Y=%lu UV=%lu) width=%u height=%u frame=%lu\n",
            (unsigned long)input_size, (unsigned long)output_size,
            (unsigned long)y_plane_size, (unsigned long)uv_plane_size,
            width, height, (unsigned long)ctx.frame_count);
#endif
    
    /* === Cached dmabuf imports + DMA-buf sync === */
    
    /* Input: cached import with DMA_BUF_IOCTL_SYNC to invalidate GPU cache.
     * v4l2src reuses a ring of ~6 dmabuf fds — the same fd carries different
     * frame content each time. The Vulkan buffer/memory import is cached (same
     * physical pages), but we must tell the kernel to invalidate caches so the
     * GPU sees the capture device's latest writes.
     */
    int in_idx = input_cache_get(in_fd, input_size);
    if (in_idx < 0) {
        return -1;
    }
    ctx.active_input_idx = in_idx;
    
    /* DMA_BUF_SYNC_START(READ) — invalidate CPU/GPU caches for this dmabuf
     * so the compute shader reads the capture device's latest frame data */
    struct dma_buf_sync sync_start = { .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ };
    ioctl(in_fd, DMA_BUF_IOCTL_SYNC, &sync_start);
    
    VkBuffer in_buffer = ctx.input_cache[in_idx].buffer;
    
    /* Output: look up or import (same fd every frame — safe to cache) */
    if (output_cache_get(out_fd, output_size) != 0) {
        return -1;
    }
    VkBuffer out_buffer = ctx.cached_output.buffer;
    
    /* Record command buffer */
    VkCommandBuffer cmd = ctx.command_buffers[0];
    
    result = vkResetCommandPool(ctx.device, ctx.command_pool, 0);
    VK_CHECK(result, "vkResetCommandPool");
    
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    
    result = vkBeginCommandBuffer(cmd, &begin_info);
    VK_CHECK(result, "vkBeginCommandBuffer");
    
    /* Update descriptor set */
    VkDescriptorBufferInfo buffer_infos[] = {
        {in_buffer, 0, input_size},
        {out_buffer, 0, y_plane_size},
        {out_buffer, y_plane_size, uv_plane_size},
    };
    
    VkWriteDescriptorSet writes[] = {
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ctx.descriptor_sets[0],
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_infos[0],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ctx.descriptor_sets[0],
            .dstBinding = 1,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_infos[1],
        },
        {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ctx.descriptor_sets[0],
            .dstBinding = 2,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_infos[2],
        },
    };
    
    vkUpdateDescriptorSets(ctx.device, 3, writes, 0, NULL);
    
    /* Bind pipeline and descriptors */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline_layout,
                            0, 1, &ctx.descriptor_sets[0], 0, NULL);
    
    /* Push constants: width, height, pairs_per_row */
    uint32_t pairs_per_row = width / 2;
    uint32_t push_data[] = {width, height, pairs_per_row};
    vkCmdPushConstants(cmd, ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push_data), push_data);
    
    /* 2D dispatch: x = pairs per row, y = rows */
    uint32_t groups_x = (pairs_per_row + 127) / 128;  /* 128 threads per workgroup */
    uint32_t groups_y = height;
    vkCmdDispatch(cmd, groups_x, groups_y, 1);
    
    /* Memory barrier */
    VkMemoryBarrier barrier = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
        .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
        .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_HOST_READ_BIT,
    };
    
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                         0, 1, &barrier, 0, NULL, 0, NULL);
    
    result = vkEndCommandBuffer(cmd);
    VK_CHECK(result, "vkEndCommandBuffer");
    
    /* Submit */
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    
    result = vkQueueSubmit(ctx.compute_queue, 1, &submit_info, ctx.fences[0]);
    VK_CHECK(result, "vkQueueSubmit");
    
    /* Track pending state for the wait call */
    ctx.pending_in_fd = in_fd;
    ctx.has_pending = 1;
    
    return 0;
}

/* =============================================================================
 * Wait for previously submitted GPU work to complete
 * ============================================================================= */

int yuv422_vulkan_convert_wait(void) {
    if (!ctx.initialized) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "Vulkan not initialized");
        return -1;
    }
    
    if (!ctx.has_pending) {
        /* Nothing to wait for */
        return 0;
    }
    
    VkResult result = vkWaitForFences(ctx.device, 1, &ctx.fences[0], VK_TRUE, 5000000000ULL);
    
    /* DMA_BUF_SYNC_END(READ) — release our read access to the input dmabuf
     * so the capture device can write new frame data to it */
    if (ctx.pending_in_fd >= 0) {
        struct dma_buf_sync sync_end = { .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ };
        ioctl(ctx.pending_in_fd, DMA_BUF_IOCTL_SYNC, &sync_end);
    }
    
    ctx.has_pending = 0;
    ctx.pending_in_fd = -1;
    
    if (result != VK_SUCCESS) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "vkWaitForFences failed: %d (frame %lu)", result, (unsigned long)ctx.frame_count);
        fprintf(stderr, "[VULKAN-ERR] %s\n", ctx.last_error);
        return -1;
    }
    
    vkResetFences(ctx.device, 1, &ctx.fences[0]);
    ctx.frame_count++;
    
    return 0;
}

/* =============================================================================
 * Synchronous Conversion (Blocking) — convenience wrapper
 * ============================================================================= */

int yuv422_vulkan_convert_dmabuf(int in_fd, int out_fd, uint32_t width, uint32_t height) {
    double t_start = now_ms();
    
    int ret = yuv422_vulkan_convert_submit(in_fd, out_fd, width, height);
    if (ret != 0) return ret;
    
    ret = yuv422_vulkan_convert_wait();
    if (ret != 0) return ret;
    
#if VULKAN_DEBUG
    double elapsed = now_ms() - t_start;
    fprintf(stderr, "[VULKAN-PROF] Convert frame %lu: %.3f ms (SYNC)\n", (unsigned long)(ctx.frame_count - 1), elapsed);
#endif
    
    return 0;
}

/* =============================================================================
 * Release output (called by encoder after it's done)
 * With caching, this is now a no-op — we keep the import alive.
 * Only called at cleanup to actually destroy resources.
 * ============================================================================= */

int yuv422_vulkan_release_output(void) {
    if (!ctx.initialized) {
        return -1;
    }
    /* With cached imports, output stays alive — nothing to release per-frame */
    return 0;
}

/* =============================================================================
 * Asynchronous Conversion (Non-Blocking) — Stub
 * ============================================================================= */

int yuv422_vulkan_convert_async(int in_fd, int out_fd, uint32_t width, uint32_t height,
                                 int slot, int *fence_fd) {
    if (!ctx.initialized) {
        return -1;
    }
    
    if (slot < 0 || slot >= MAX_FRAMES_IN_FLIGHT) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "Invalid slot %d", slot);
        return -1;
    }
    
    if (ctx.slot_busy[slot]) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "Slot %d is busy", slot);
        return -1;
    }
    
    ctx.slot_busy[slot] = 1;
    
    snprintf(ctx.last_error, sizeof(ctx.last_error), "Async conversion not yet implemented");
    return -1;
}

int yuv422_vulkan_wait_async(int slot, int timeout_ms) {
    if (!ctx.initialized || slot < 0 || slot >= MAX_FRAMES_IN_FLIGHT) {
        return -1;
    }
    
    if (!ctx.slot_busy[slot]) {
        return 0;
    }
    
    VkResult result = vkWaitForFences(ctx.device, 1, &ctx.fences[slot], VK_TRUE,
                                       timeout_ms * 1000000ULL);
    
    if (result == VK_TIMEOUT) {
        return -1;
    }
    
    if (result == VK_SUCCESS) {
        ctx.slot_busy[slot] = 0;
        return 0;
    }
    
    return -1;
}

/* =============================================================================
 * Cleanup
 * ============================================================================= */

void yuv422_vulkan_cleanup(void) {
    if (!ctx.initialized) {
        return;
    }
    
    vkDeviceWaitIdle(ctx.device);
    
    /* Destroy cached imports */
    for (int i = 0; i < ctx.input_cache_count; i++) {
        cache_entry_destroy(&ctx.input_cache[i]);
    }
    ctx.input_cache_count = 0;
    
    cache_entry_destroy(&ctx.cached_output);
    
    /* Destroy Vulkan resources */
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroyFence(ctx.device, ctx.fences[i], NULL);
        vkDestroySemaphore(ctx.device, ctx.semaphores[i], NULL);
    }
    
    if (ctx.pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(ctx.device, ctx.pipeline, NULL);
    if (ctx.pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(ctx.device, ctx.pipeline_layout, NULL);
    if (ctx.shader_module != VK_NULL_HANDLE)
        vkDestroyShaderModule(ctx.device, ctx.shader_module, NULL);
    if (ctx.descriptor_set_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(ctx.device, ctx.descriptor_set_layout, NULL);
    if (ctx.descriptor_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(ctx.device, ctx.descriptor_pool, NULL);
    if (ctx.command_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(ctx.device, ctx.command_pool, NULL);
    if (ctx.device != VK_NULL_HANDLE)
        vkDestroyDevice(ctx.device, NULL);
    if (ctx.instance != VK_NULL_HANDLE)
        vkDestroyInstance(ctx.instance, NULL);
    
    memset(&ctx, 0, sizeof(ctx));
    fprintf(stderr, "[VULKAN] Cleanup complete\n");
}

const char *yuv422_vulkan_last_error(void) {
    return ctx.last_error;
}
