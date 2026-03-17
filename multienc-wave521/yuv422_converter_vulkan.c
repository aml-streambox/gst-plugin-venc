/*
 * Vulkan-based Zero-Copy YUV422 to P010 Converter
 * 
 * This implementation uses Vulkan compute shaders for high-performance
 * color format conversion with zero CPU memory copies.
 * 
 * Key Features:
 * - VK_EXT_external_memory_dma_buf for dmabuf import/export
 * - Compute shaders for mathematical color conversion
 * - Triple-buffered async pipeline for 4K60 performance
 * - Timeline semaphores for GPU→Encoder synchronization
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
#include <errno.h>

#include "yuv422_converter_vulkan.h"

/* =============================================================================
 * Configuration Constants
 * ============================================================================= */

#define MAX_FRAMES_IN_FLIGHT 3
#define VK_CHECK(result, msg) do { \
    if (result != VK_SUCCESS) { \
        snprintf(ctx.last_error, sizeof(ctx.last_error), "%s: %d", msg, result); \
        fprintf(stderr, "[VULKAN-ERR] %s\n", ctx.last_error); \
        return -1; \
    } \
} while(0)

/* =============================================================================
 * Shader Code (SPIR-V will be embedded or compiled at runtime)
 * ============================================================================= */

/*
 * Compute shader for YUV422→P010 conversion
 * 
 * Input: 40-bit packed YUV422 (5 bytes per 2 pixels)
 * Output: P010_10LE format (Y plane + UV plane, MSB-aligned, native LE — no byte swap)
 * 
 * Each workgroup processes 256 pixels (128 work-items)
 * Grid size: (3840*2160/2 + 255) / 256 workgroups for 4K
 */
static const char *compute_shader_glsl = 
    "#version 450\n"
    "#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require\n"
    "#extension GL_EXT_shader_8bit_storage : require\n"
    "\n"
    "layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;\n"
    "\n"
    "layout(set = 0, binding = 0) readonly buffer InputBuffer {\n"
    "    uint8_t data[];\n"
    "} input_buffer;\n"
    "\n"
    "layout(set = 0, binding = 1) writeonly buffer YOutputBuffer {\n"
    "    uint16_t data[];\n"
    "} y_output;\n"
    "\n"
    "layout(set = 0, binding = 2) writeonly buffer UVOutputBuffer {\n"
    "    uint16_t data[];\n"
    "} uv_output;\n"
    "\n"
    "layout(push_constant) uniform Params {\n"
    "    uint width;\n"
    "    uint height;\n"
    "} params;\n"
    "\n"
    "void main() {\n"
    "    uint pair_idx = gl_GlobalInvocationID.x;\n"
    "    uint total_pairs = params.width * params.height / 2;\n"
    "    \n"
    "    if (pair_idx >= total_pairs) return;\n"
    "    \n"
    "    uint byte_offset = pair_idx * 5;\n"
    "    \n"
    "    uint b0 = input_buffer.data[byte_offset + 0];\n"
    "    uint b1 = input_buffer.data[byte_offset + 1];\n"
    "    uint b2 = input_buffer.data[byte_offset + 2];\n"
    "    uint b3 = input_buffer.data[byte_offset + 3];\n"
    "    uint b4 = input_buffer.data[byte_offset + 4];\n"
    "    \n"
    "    uint u0 = (b0) | ((b1 & 0x03) << 8);\n"
    "    uint y0 = ((b1 >> 2) & 0x3F) | ((b2 & 0x0F) << 6);\n"
    "    uint v0 = ((b2 >> 4) & 0x0F) | ((b3 & 0x3F) << 4);\n"
    "    uint y1 = ((b3 >> 6) & 0x03) | (b4 << 2);\n"
    "    \n"
    "    // P010_10LE: MSB-aligned 10-bit, no byte swap on ARM\n"
    "    uint16_t y0_p010 = uint16_t(y0 << 6);\n"
    "    uint16_t y1_p010 = uint16_t(y1 << 6);\n"
    "    \n"
    "    uint y_idx = pair_idx * 2;\n"
    "    y_output.data[y_idx + 0] = y0_p010;\n"
    "    y_output.data[y_idx + 1] = y1_p010;\n"
    "    \n"
    "    uint row = pair_idx / (params.width / 2);\n"
    "    uint col = pair_idx % (params.width / 2);\n"
    "    \n"
    "    if ((row & 1) == 0 && row + 1 < params.height) {\n"
    "        uint next_offset = byte_offset + (params.width * 5 / 2);\n"
    "        uint nb0 = input_buffer.data[next_offset + 0];\n"
    "        uint nb1 = input_buffer.data[next_offset + 1];\n"
    "        uint nb2 = input_buffer.data[next_offset + 2];\n"
    "        \n"
    "        uint u1 = (nb0) | ((nb1 & 0x03) << 8);\n"
    "        uint v1 = ((nb2 >> 4) & 0x0F) | ((uint(input_buffer.data[next_offset + 3]) & 0x3F) << 4);\n"
    "        \n"
    "        uint u_avg = (u0 + u1 + 1) >> 1;\n"
    "        uint v_avg = (v0 + v1 + 1) >> 1;\n"
    "        \n"
    "        uint16_t u_p010 = uint16_t(u_avg << 6);\n"
    "        uint16_t v_p010 = uint16_t(v_avg << 6);\n"
    "        \n"
    "        uint uv_row = row / 2;\n"
    "        uint uv_idx = (uv_row * (params.width / 2) + col) * 2;\n"
    "        uv_output.data[uv_idx + 0] = u_p010;\n"
    "        uv_output.data[uv_idx + 1] = v_p010;\n"
    "    }\n"
    "}\n";


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
    int frame_count;  /* Track frames for fence management */
    char last_error[256];
    
    /* Output resource tracking for zero-copy
     * We keep output VkDeviceMemory alive until encoder is done
     */
    struct {
        VkBuffer buffer;
        VkDeviceMemory memory;
        int valid;  /* 1 if we have an output to release */
    } output_resource;
    
    /* Input resource tracking (freed after compute completes) */
    struct {
        VkBuffer buffer;
        VkDeviceMemory memory;
        int valid;
    } input_resource;
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

/* Find memory type index for device-local or host-visible memory */
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

/* Create shader module from embedded SPIR-V */
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
 * Initialization
 * ============================================================================= */

int yuv422_vulkan_init(uint32_t width, uint32_t height) {
    if (ctx.initialized) {
        return 0;
    }
    
    ctx.width = width;
    ctx.height = height;
    
    /* Check for Vulkan support on target */
    uint32_t instance_version = 0;
    VkResult result = vkEnumerateInstanceVersion(&instance_version);
    if (result != VK_SUCCESS) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), 
                "Vulkan not supported on this system");
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
    
    /* Enable external memory extensions */
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
    
    /* Select first device that supports compute */
    for (uint32_t i = 0; i < device_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        
        /* Check for compute queue */
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
        
        if (ctx.physical_device != VK_NULL_HANDLE) {
            break;
        }
    }
    
    free(devices);
    
    if (ctx.physical_device == VK_NULL_HANDLE) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "No device with compute support");
        return -1;
    }
    
    /* Get memory properties */
    vkGetPhysicalDeviceMemoryProperties(ctx.physical_device, &ctx.memory_props);
    
    /* Create logical device */
    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .queueFamilyIndex = ctx.compute_queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };
    
    /* Enable external memory extensions */
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
    
    /* Get compute queue */
    vkGetDeviceQueue(ctx.device, ctx.compute_queue_family, 0, &ctx.compute_queue);
    
    /* Create command pool with TRANSIENT_BIT for short-lived command buffers
     * (optimization for Mali - tells driver buffers are short-lived) */
    VkCommandPoolCreateInfo pool_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
        .queueFamilyIndex = ctx.compute_queue_family,
        .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT |
                 VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
    };
    
    result = vkCreateCommandPool(ctx.device, &pool_info, NULL, &ctx.command_pool);
    VK_CHECK(result, "vkCreateCommandPool");
    
    /* Allocate command buffers for triple buffering */
    VkCommandBufferAllocateInfo cmd_alloc = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .commandPool = ctx.command_pool,
        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
    };
    
    result = vkAllocateCommandBuffers(ctx.device, &cmd_alloc, ctx.command_buffers);
    VK_CHECK(result, "vkAllocateCommandBuffers");
    
    /* Create fences for synchronization */
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
    
    /* Create pipeline layout */
    VkPushConstantRange push_constant = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = sizeof(uint32_t) * 2,  // width, height
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
    
    /* Compile shader */
    /* 
     * TODO: Load pre-compiled SPIR-V binary
     * The shader should be compiled offline and embedded as a byte array
     */
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
    ctx.output_resource.valid = 0;
    ctx.input_resource.valid = 0;
    fprintf(stderr, "[VULKAN] Initialization complete\n");
    
    return 0;
}

/* =============================================================================
 * DMA-BUF Import/Export
 * ============================================================================= */

/* Import dmabuf as Vulkan buffer.
 * IMPORTANT: This function TAKES OWNERSHIP of 'fd' — Vulkan will close it.
 * Caller must pass a dup()'d fd if they need to keep the original.
 */
static int import_dmabuf(int fd, VkDeviceSize size, VkBuffer *buffer, VkDeviceMemory *memory) {
    /* Create buffer with external memory handle */
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
    
    /* Get memory requirements */
    VkMemoryRequirements mem_reqs;
    vkGetBufferMemoryRequirements(ctx.device, *buffer, &mem_reqs);
    
    /* Import dmabuf as device memory — Vulkan takes ownership of fd */
    VkImportMemoryFdInfoKHR import_info = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        .fd = fd,
    };
    
    /* Use the actual dmabuf size (from fstat) if larger than mem_reqs.size
     * This ensures Vulkan knows about the full allocation */
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
        /* fd is consumed by vkAllocateMemory even on failure per spec */
        return -1;
    }
    
    /* Bind memory to buffer */
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
 * Synchronous Conversion (Blocking)
 * ============================================================================= */

/* Release previous output resources before importing new ones */
static void release_previous_output(void) {
    if (ctx.output_resource.valid) {
        fprintf(stderr, "[VULKAN] Releasing previous output resources\n");
        vkDestroyBuffer(ctx.device, ctx.output_resource.buffer, NULL);
        vkFreeMemory(ctx.device, ctx.output_resource.memory, NULL);
        ctx.output_resource.valid = 0;
    }
}

/* Release previous input resources */
static void release_previous_input(void) {
    if (ctx.input_resource.valid) {
        vkDestroyBuffer(ctx.device, ctx.input_resource.buffer, NULL);
        vkFreeMemory(ctx.device, ctx.input_resource.memory, NULL);
        ctx.input_resource.valid = 0;
    }
}

int yuv422_vulkan_release_output(void) {
    if (!ctx.initialized) {
        return -1;
    }
    
    if (!ctx.output_resource.valid) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "No output to release");
        return -1;
    }
    
    release_previous_output();
    return 0;
}

int yuv422_vulkan_convert_dmabuf(int in_fd, int out_fd, uint32_t width, uint32_t height) {
    if (!ctx.initialized) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "Vulkan not initialized");
        return -1;
    }
    
    double t_start = now_ms();
    VkResult result;
    
    /* === Fence lifecycle ===
     * Frame 0: fence starts SIGNALED (created with SIGNALED_BIT).
     *   We reset it before submit.
     * Frame 1+: fence was reset after WaitForFences at end of previous frame.
     *   It's already UNSIGNALED, ready for submit.
     *
     * Before reusing command pool / buffers, we must ensure previous GPU work is done.
     * For synchronous path, we already waited at end of previous frame, so it's safe.
     * But add an explicit wait anyway for safety (returns immediately if already signaled/reset).
     */
    if (ctx.frame_count == 0) {
        /* First frame: fence starts signaled, reset it */
        vkResetFences(ctx.device, 1, &ctx.fences[0]);
    }
    /* For frame_count > 0, fence is already unsignaled (was reset after last WaitForFences) */
    
    /* Release any previous input resources (safe - GPU idle from previous WaitForFences) */
    release_previous_input();
    
    /* Release any previous output resources */
    release_previous_output();
    
    /* Calculate buffer sizes */
    VkDeviceSize input_size = (VkDeviceSize)width * height * 5 / 2;   // 40-bit packed YUV422
    VkDeviceSize y_plane_size = (VkDeviceSize)width * height * 2;     // P010 Y plane (16-bit per pixel)
    VkDeviceSize uv_plane_size = (VkDeviceSize)width * height;        // P010 UV plane (width*height/2 * 2 bytes * 2 components)
    VkDeviceSize output_size = y_plane_size + uv_plane_size;          // Total P010 = width*height*3
    
    /* Validate dmabuf sizes with fstat */
    struct stat in_stat, out_stat;
    if (fstat(in_fd, &in_stat) == 0) {
        fprintf(stderr, "[VULKAN] Input dmabuf fd=%d actual_size=%ld expected=%lu\n",
                in_fd, (long)in_stat.st_size, (unsigned long)input_size);
        if ((VkDeviceSize)in_stat.st_size > 0 && (VkDeviceSize)in_stat.st_size < input_size) {
            fprintf(stderr, "[VULKAN-WARN] Input dmabuf smaller than expected!\n");
        }
    }
    if (fstat(out_fd, &out_stat) == 0) {
        fprintf(stderr, "[VULKAN] Output dmabuf fd=%d actual_size=%ld expected=%lu\n",
                out_fd, (long)out_stat.st_size, (unsigned long)output_size);
        if ((VkDeviceSize)out_stat.st_size > 0 && (VkDeviceSize)out_stat.st_size < output_size) {
            fprintf(stderr, "[VULKAN-WARN] Output dmabuf smaller than expected! May cause GPU crash.\n");
            /* Use actual dmabuf size if it's smaller to avoid OOB */
            if ((VkDeviceSize)out_stat.st_size > 0) {
                output_size = out_stat.st_size;
                /* Recalculate plane sizes based on actual buffer size */
                if (output_size >= y_plane_size) {
                    uv_plane_size = output_size - y_plane_size;
                } else {
                    snprintf(ctx.last_error, sizeof(ctx.last_error),
                             "Output dmabuf too small for Y plane: %lu < %lu",
                             (unsigned long)output_size, (unsigned long)y_plane_size);
                    return -1;
                }
            }
        }
    }
    
    fprintf(stderr, "[VULKAN] Buffer sizes: input=%lu output=%lu (Y=%lu UV=%lu) width=%u height=%u frame=%d\n",
            (unsigned long)input_size, (unsigned long)output_size,
            (unsigned long)y_plane_size, (unsigned long)uv_plane_size,
            width, height, ctx.frame_count);
    
    /* Import INPUT dmabuf into Vulkan */
    VkBuffer in_buffer;
    VkDeviceMemory in_memory;
    
    /* dup() before passing to import_dmabuf which takes ownership */
    int in_fd_dup = dup(in_fd);
    if (in_fd_dup < 0) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "Failed to dup input fd: %s", strerror(errno));
        return -1;
    }
    
    if (import_dmabuf(in_fd_dup, input_size, &in_buffer, &in_memory) != 0) {
        /* import_dmabuf closes/consumes fd on failure, don't close again */
        return -1;
    }
    
    /* Store input resources for cleanup after GPU completes */
    ctx.input_resource.buffer = in_buffer;
    ctx.input_resource.memory = in_memory;
    ctx.input_resource.valid = 1;
    
    /* Import OUTPUT dmabuf into Vulkan (KEEP ALIVE until encoder done) */
    VkBuffer out_buffer;
    VkDeviceMemory out_memory;
    
    /* dup() before passing to import_dmabuf which takes ownership */
    int out_fd_dup = dup(out_fd);
    if (out_fd_dup < 0) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "Failed to dup output fd: %s", strerror(errno));
        release_previous_input();
        return -1;
    }
    
    if (import_dmabuf(out_fd_dup, output_size, &out_buffer, &out_memory) != 0) {
        /* import_dmabuf closes/consumes fd on failure */
        release_previous_input();
        return -1;
    }
    
    /* Store output resource for later release */
    ctx.output_resource.buffer = out_buffer;
    ctx.output_resource.memory = out_memory;
    ctx.output_resource.valid = 1;
    
    /* Record command buffer */
    VkCommandBuffer cmd = ctx.command_buffers[0];
    
    /* Reset command pool — safe because previous GPU work completed (synchronous path) */
    result = vkResetCommandPool(ctx.device, ctx.command_pool, 0);
    VK_CHECK(result, "vkResetCommandPool");
    
    VkCommandBufferBeginInfo begin_info = {
        .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
    };
    
    result = vkBeginCommandBuffer(cmd, &begin_info);
    VK_CHECK(result, "vkBeginCommandBuffer");
    
    /* Update descriptor set - input at binding 0, Y output at binding 1, UV output at binding 2 */
    VkDescriptorBufferInfo buffer_infos[] = {
        {in_buffer, 0, input_size},                                // Binding 0: Input packed YUV422
        {out_buffer, 0, y_plane_size},                             // Binding 1: Output Y plane
        {out_buffer, y_plane_size, uv_plane_size},                 // Binding 2: Output UV plane
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
    
    /* Bind pipeline and descriptor set */
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, ctx.pipeline_layout,
                            0, 1, &ctx.descriptor_sets[0], 0, NULL);
    
    /* Push constants */
    uint32_t push_data[] = {width, height};
    vkCmdPushConstants(cmd, ctx.pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push_data), push_data);
    
    /* Dispatch compute shader */
    uint32_t pair_count = width * height / 2;
    uint32_t group_count = (pair_count + 127) / 128;  // 128 threads per group
    vkCmdDispatch(cmd, group_count, 1, 1);
    
    /* Memory barrier to ensure GPU writes are visible */
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
    
    /* Submit with fence */
    VkSubmitInfo submit_info = {
        .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd,
    };
    
    result = vkQueueSubmit(ctx.compute_queue, 1, &submit_info, ctx.fences[0]);
    VK_CHECK(result, "vkQueueSubmit");
    
    /* Wait for GPU completion (synchronous path) */
    result = vkWaitForFences(ctx.device, 1, &ctx.fences[0], VK_TRUE, 5000000000ULL);
    if (result != VK_SUCCESS) {
        snprintf(ctx.last_error, sizeof(ctx.last_error), "vkWaitForFences failed: %d (frame %d)", result, ctx.frame_count);
        fprintf(stderr, "[VULKAN-ERR] %s\n", ctx.last_error);
        return -1;
    }
    
    /* Reset fence immediately after wait — ready for next submit */
    vkResetFences(ctx.device, 1, &ctx.fences[0]);
    
    /* Free INPUT resources immediately (GPU is done reading) */
    release_previous_input();
    
    /* NOTE: OUTPUT resources are NOT freed here!
     * They remain valid in ctx.output_resource until yuv422_vulkan_release_output() is called
     * This ensures the output dmabuf fd remains valid for the encoder
     */
    
    ctx.frame_count++;
    double elapsed = now_ms() - t_start;
    fprintf(stderr, "[VULKAN-PROF] Convert frame %d: %.3f ms (ZERO-COPY)\n", ctx.frame_count - 1, elapsed);
    
    return 0;
}

/* =============================================================================
 * Asynchronous Conversion (Non-Blocking)
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
    
    /* Mark slot as busy */
    ctx.slot_busy[slot] = 1;
    
    /* TODO: Implement async version with:
     * 1. Import dmabufs
     * 2. Record command buffer for this slot
     * 3. Submit with signal semaphore
     * 4. Export semaphore as sync fence fd
     * 5. Encoder can poll() on fence_fd
     */
    
    snprintf(ctx.last_error, sizeof(ctx.last_error), "Async conversion not yet implemented");
    return -1;
}

int yuv422_vulkan_wait_async(int slot, int timeout_ms) {
    if (!ctx.initialized || slot < 0 || slot >= MAX_FRAMES_IN_FLIGHT) {
        return -1;
    }
    
    if (!ctx.slot_busy[slot]) {
        return 0;  // Already done
    }
    
    /* Wait on fence */
    VkResult result = vkWaitForFences(ctx.device, 1, &ctx.fences[slot], VK_TRUE,
                                      timeout_ms * 1000000ULL);  // Convert to ns
    
    if (result == VK_TIMEOUT) {
        return -1;  // Timeout
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
    
    /* Wait for all operations to complete */
    vkDeviceWaitIdle(ctx.device);
    
    /* Release tracked resources */
    release_previous_input();
    release_previous_output();
    
    /* Destroy resources */
    for (int i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        vkDestroyFence(ctx.device, ctx.fences[i], NULL);
        vkDestroySemaphore(ctx.device, ctx.semaphores[i], NULL);
    }
    
    if (ctx.pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(ctx.device, ctx.pipeline, NULL);
    }
    if (ctx.pipeline_layout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(ctx.device, ctx.pipeline_layout, NULL);
    }
    if (ctx.shader_module != VK_NULL_HANDLE) {
        vkDestroyShaderModule(ctx.device, ctx.shader_module, NULL);
    }
    if (ctx.descriptor_set_layout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(ctx.device, ctx.descriptor_set_layout, NULL);
    }
    if (ctx.descriptor_pool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(ctx.device, ctx.descriptor_pool, NULL);
    }
    if (ctx.command_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(ctx.device, ctx.command_pool, NULL);
    }
    if (ctx.device != VK_NULL_HANDLE) {
        vkDestroyDevice(ctx.device, NULL);
    }
    if (ctx.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(ctx.instance, NULL);
    }
    
    memset(&ctx, 0, sizeof(ctx));
    fprintf(stderr, "[VULKAN] Cleanup complete\n");
}

const char *yuv422_vulkan_last_error(void) {
    return ctx.last_error;
}
