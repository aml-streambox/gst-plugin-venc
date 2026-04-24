/*
 * Test-only Vulkan pass: repack standard P010 (val << 6) into the Wave521
 * byte-swapped layout on a single contiguous dmabuf in place.
 */

#define _GNU_SOURCE
#include <errno.h>
#include <linux/dma-buf.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

#include "p010_repack_vulkan.h"

struct P010RepackCtx {
    VkInstance instance;
    VkPhysicalDevice physical_device;
    VkDevice device;
    VkQueue compute_queue;
    uint32_t compute_queue_family;
    VkPhysicalDeviceMemoryProperties memory_props;
    VkCommandPool command_pool;
    VkCommandBuffer command_buffer;
    VkFence fence;
    VkDescriptorPool descriptor_pool;
    VkDescriptorSetLayout descriptor_set_layout;
    VkDescriptorSet descriptor_set;
    VkPipelineLayout pipeline_layout;
    VkShaderModule shader_module;
    VkPipeline pipeline;
    char last_error[256];
    int initialized;
};

#include "p010_repack_wave_spv.h"

#define P010_VK_FAIL(ctx, fmt, ...) \
    do { \
        snprintf((ctx)->last_error, sizeof((ctx)->last_error), fmt, ##__VA_ARGS__); \
        return -1; \
    } while (0)

#define P010_VK_INIT_FAIL(ctx, fmt, ...) \
    do { \
        snprintf((ctx)->last_error, sizeof((ctx)->last_error), fmt, ##__VA_ARGS__); \
        p010_repack_vulkan_cleanup(ctx); \
        return NULL; \
    } while (0)

static int
find_memory_type(P010RepackCtx *ctx, uint32_t type_filter, VkMemoryPropertyFlags props)
{
    uint32_t i;

    for (i = 0; i < ctx->memory_props.memoryTypeCount; i++) {
        if ((type_filter & (1u << i)) &&
            (ctx->memory_props.memoryTypes[i].propertyFlags & props) == props)
            return (int)i;
    }

    return -1;
}

static int
import_dmabuf(P010RepackCtx *ctx, int fd, VkDeviceSize size, VkBuffer *buffer, VkDeviceMemory *memory)
{
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
    VkMemoryRequirements mem_reqs;
    VkImportMemoryFdInfoKHR import_info = {
        .sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR,
        .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT,
        .fd = fd,
    };
    VkMemoryAllocateInfo alloc_info = {
        .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
        .pNext = &import_info,
    };
    VkResult result;
    int mem_type;

    result = vkCreateBuffer(ctx->device, &buffer_info, NULL, buffer);
    if (result != VK_SUCCESS) {
        close(fd);
        P010_VK_FAIL(ctx, "vkCreateBuffer failed: %d", result);
    }

    vkGetBufferMemoryRequirements(ctx->device, *buffer, &mem_reqs);
    alloc_info.allocationSize = mem_reqs.size;
    mem_type = find_memory_type(ctx, mem_reqs.memoryTypeBits, 0);
    if (mem_type < 0) {
        vkDestroyBuffer(ctx->device, *buffer, NULL);
        close(fd);
        P010_VK_FAIL(ctx, "No suitable memory type for dmabuf import");
    }
    alloc_info.memoryTypeIndex = (uint32_t) mem_type;

    result = vkAllocateMemory(ctx->device, &alloc_info, NULL, memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(ctx->device, *buffer, NULL);
        close(fd);
        P010_VK_FAIL(ctx, "vkAllocateMemory failed: %d", result);
    }

    result = vkBindBufferMemory(ctx->device, *buffer, *memory, 0);
    if (result != VK_SUCCESS) {
        vkFreeMemory(ctx->device, *memory, NULL);
        vkDestroyBuffer(ctx->device, *buffer, NULL);
        P010_VK_FAIL(ctx, "vkBindBufferMemory failed: %d", result);
    }

    return 0;
}

P010RepackCtx *
p010_repack_vulkan_init(void)
{
    P010RepackCtx *ctx = calloc(1, sizeof(*ctx));
    VkResult result;
    uint32_t device_count = 0;
    float queue_priority = 1.0f;
    VkApplicationInfo app_info = {
        .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pApplicationName = "P010Repack",
        .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
        .pEngineName = "VulkanCompute",
        .engineVersion = VK_MAKE_VERSION(1, 0, 0),
        .apiVersion = VK_API_VERSION_1_2,
    };
    const char *instance_extensions[] = {
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME,
    };
    VkInstanceCreateInfo instance_info = {
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pApplicationInfo = &app_info,
        .enabledExtensionCount = 2,
        .ppEnabledExtensionNames = instance_extensions,
    };
    const char *device_extensions[] = {
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
    };
    VkDescriptorSetLayoutBinding binding = {
        .binding = 0,
        .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 1,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
    };
    VkDescriptorSetAllocateInfo desc_alloc;
    VkCommandBufferAllocateInfo cmd_alloc;

    if (!ctx)
        return NULL;

    result = vkCreateInstance(&instance_info, NULL, &ctx->instance);
    if (result != VK_SUCCESS)
        P010_VK_INIT_FAIL(ctx, "vkCreateInstance failed: %d", result);

    vkEnumeratePhysicalDevices(ctx->instance, &device_count, NULL);
    if (device_count == 0)
        P010_VK_INIT_FAIL(ctx, "No Vulkan devices found");

    {
        VkPhysicalDevice *devices = calloc(device_count, sizeof(*devices));
        uint32_t i;

        if (!devices)
            P010_VK_INIT_FAIL(ctx, "Failed to allocate device list");

        vkEnumeratePhysicalDevices(ctx->instance, &device_count, devices);
        for (i = 0; i < device_count && ctx->physical_device == VK_NULL_HANDLE; i++) {
            uint32_t queue_family_count = 0;
            VkQueueFamilyProperties *queue_families;
            uint32_t j;

            vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queue_family_count, NULL);
            queue_families = calloc(queue_family_count, sizeof(*queue_families));
            if (!queue_families)
                continue;

            vkGetPhysicalDeviceQueueFamilyProperties(devices[i], &queue_family_count, queue_families);
            for (j = 0; j < queue_family_count; j++) {
                if (queue_families[j].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                    ctx->physical_device = devices[i];
                    ctx->compute_queue_family = j;
                    break;
                }
            }
            free(queue_families);
        }
        free(devices);
    }

    if (ctx->physical_device == VK_NULL_HANDLE)
        P010_VK_INIT_FAIL(ctx, "No compute-capable Vulkan device found");

    vkGetPhysicalDeviceMemoryProperties(ctx->physical_device, &ctx->memory_props);

    {
        VkDeviceQueueCreateInfo queue_info = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .queueFamilyIndex = ctx->compute_queue_family,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        };
        VkDeviceCreateInfo device_info = {
            .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_info,
            .enabledExtensionCount = 4,
            .ppEnabledExtensionNames = device_extensions,
        };

        result = vkCreateDevice(ctx->physical_device, &device_info, NULL, &ctx->device);
        if (result != VK_SUCCESS)
            P010_VK_INIT_FAIL(ctx, "vkCreateDevice failed: %d", result);
    }

    vkGetDeviceQueue(ctx->device, ctx->compute_queue_family, 0, &ctx->compute_queue);

    {
        VkCommandPoolCreateInfo pool_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .queueFamilyIndex = ctx->compute_queue_family,
            .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT | VK_COMMAND_POOL_CREATE_TRANSIENT_BIT,
        };
        result = vkCreateCommandPool(ctx->device, &pool_info, NULL, &ctx->command_pool);
        if (result != VK_SUCCESS)
            P010_VK_INIT_FAIL(ctx, "vkCreateCommandPool failed: %d", result);
    }

    cmd_alloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmd_alloc.pNext = NULL;
    cmd_alloc.commandPool = ctx->command_pool;
    cmd_alloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmd_alloc.commandBufferCount = 1;
    result = vkAllocateCommandBuffers(ctx->device, &cmd_alloc, &ctx->command_buffer);
    if (result != VK_SUCCESS)
        P010_VK_INIT_FAIL(ctx, "vkAllocateCommandBuffers failed: %d", result);

    {
        VkFenceCreateInfo fence_info = {
            .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .flags = VK_FENCE_CREATE_SIGNALED_BIT,
        };
        result = vkCreateFence(ctx->device, &fence_info, NULL, &ctx->fence);
        if (result != VK_SUCCESS)
            P010_VK_INIT_FAIL(ctx, "vkCreateFence failed: %d", result);
    }

    {
        VkDescriptorPoolSize pool_size = {
            .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
        };
        VkDescriptorPoolCreateInfo pool_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .maxSets = 1,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        result = vkCreateDescriptorPool(ctx->device, &pool_info, NULL, &ctx->descriptor_pool);
        if (result != VK_SUCCESS)
            P010_VK_INIT_FAIL(ctx, "vkCreateDescriptorPool failed: %d", result);
    }

    {
        VkDescriptorSetLayoutCreateInfo layout_info = {
            .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
            .bindingCount = 1,
            .pBindings = &binding,
        };
        result = vkCreateDescriptorSetLayout(ctx->device, &layout_info, NULL, &ctx->descriptor_set_layout);
        if (result != VK_SUCCESS)
            P010_VK_INIT_FAIL(ctx, "vkCreateDescriptorSetLayout failed: %d", result);
    }

    desc_alloc.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    desc_alloc.pNext = NULL;
    desc_alloc.descriptorPool = ctx->descriptor_pool;
    desc_alloc.descriptorSetCount = 1;
    desc_alloc.pSetLayouts = &ctx->descriptor_set_layout;
    result = vkAllocateDescriptorSets(ctx->device, &desc_alloc, &ctx->descriptor_set);
    if (result != VK_SUCCESS)
        P010_VK_INIT_FAIL(ctx, "vkAllocateDescriptorSets failed: %d", result);

    {
        VkPushConstantRange push_constant = {
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .offset = 0,
            .size = sizeof(uint32_t),
        };
        VkPipelineLayoutCreateInfo pipeline_layout_info = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
            .setLayoutCount = 1,
            .pSetLayouts = &ctx->descriptor_set_layout,
            .pushConstantRangeCount = 1,
            .pPushConstantRanges = &push_constant,
        };
        result = vkCreatePipelineLayout(ctx->device, &pipeline_layout_info, NULL, &ctx->pipeline_layout);
        if (result != VK_SUCCESS)
            P010_VK_INIT_FAIL(ctx, "vkCreatePipelineLayout failed: %d", result);
    }

    {
        VkShaderModuleCreateInfo create_info = {
            .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
            .codeSize = sizeof(p010_repack_wave_spv),
            .pCode = (const uint32_t *) p010_repack_wave_spv,
        };
        result = vkCreateShaderModule(ctx->device, &create_info, NULL, &ctx->shader_module);
        if (result != VK_SUCCESS)
            P010_VK_INIT_FAIL(ctx, "vkCreateShaderModule failed: %d", result);
    }

    {
        VkComputePipelineCreateInfo pipeline_info = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {
                .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                .module = ctx->shader_module,
                .pName = "main",
            },
            .layout = ctx->pipeline_layout,
        };
        result = vkCreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pipeline_info, NULL, &ctx->pipeline);
        if (result != VK_SUCCESS)
            P010_VK_INIT_FAIL(ctx, "vkCreateComputePipelines failed: %d", result);
    }

    ctx->initialized = 1;
    return ctx;
}

int
p010_repack_vulkan_convert_inplace(P010RepackCtx *ctx, int fd, uint32_t width, uint32_t height)
{
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkResult result;
    VkDeviceSize size = (VkDeviceSize) width * height * 3;
    uint32_t total_words = (uint32_t) (size / 4);
    int fd_dup;
    struct dma_buf_sync sync_start = { .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE };
    struct dma_buf_sync sync_end = { .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ | DMA_BUF_SYNC_WRITE };

    if (!ctx || !ctx->initialized)
        return -1;

    fd_dup = dup(fd);
    if (fd_dup < 0)
        P010_VK_FAIL(ctx, "dup(%d) failed: %s", fd, strerror(errno));

    if (import_dmabuf(ctx, fd_dup, size, &buffer, &memory) != 0)
        return -1;

    ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync_start);

    result = vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS)
        goto fail;
    result = vkResetFences(ctx->device, 1, &ctx->fence);
    if (result != VK_SUCCESS)
        goto fail;

    {
        VkDescriptorBufferInfo buffer_info = { buffer, 0, size };
        VkWriteDescriptorSet write = {
            .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstSet = ctx->descriptor_set,
            .dstBinding = 0,
            .descriptorCount = 1,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pBufferInfo = &buffer_info,
        };
        vkUpdateDescriptorSets(ctx->device, 1, &write, 0, NULL);
    }

    result = vkResetCommandPool(ctx->device, ctx->command_pool, 0);
    if (result != VK_SUCCESS)
        goto fail;

    {
        VkCommandBufferBeginInfo begin_info = {
            .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        };
        result = vkBeginCommandBuffer(ctx->command_buffer, &begin_info);
        if (result != VK_SUCCESS)
            goto fail;
    }

    vkCmdBindPipeline(ctx->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, ctx->pipeline);
    vkCmdBindDescriptorSets(ctx->command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                            ctx->pipeline_layout, 0, 1, &ctx->descriptor_set, 0, NULL);
    vkCmdPushConstants(ctx->command_buffer, ctx->pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(total_words), &total_words);
    vkCmdDispatch(ctx->command_buffer, (total_words + 255u) / 256u, 1, 1);

    {
        VkMemoryBarrier barrier = {
            .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_HOST_READ_BIT,
        };
        vkCmdPipelineBarrier(ctx->command_buffer,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                             0, 1, &barrier, 0, NULL, 0, NULL);
    }

    result = vkEndCommandBuffer(ctx->command_buffer);
    if (result != VK_SUCCESS)
        goto fail;

    {
        VkSubmitInfo submit_info = {
            .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .commandBufferCount = 1,
            .pCommandBuffers = &ctx->command_buffer,
        };
        result = vkQueueSubmit(ctx->compute_queue, 1, &submit_info, ctx->fence);
        if (result != VK_SUCCESS)
            goto fail;
    }

    result = vkWaitForFences(ctx->device, 1, &ctx->fence, VK_TRUE, UINT64_MAX);
    if (result != VK_SUCCESS)
        goto fail;

    ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync_end);
    vkDestroyBuffer(ctx->device, buffer, NULL);
    vkFreeMemory(ctx->device, memory, NULL);
    return 0;

fail:
    snprintf(ctx->last_error, sizeof(ctx->last_error), "Vulkan repack failed: %d", result);
    ioctl(fd, DMA_BUF_IOCTL_SYNC, &sync_end);
    if (buffer != VK_NULL_HANDLE)
        vkDestroyBuffer(ctx->device, buffer, NULL);
    if (memory != VK_NULL_HANDLE)
        vkFreeMemory(ctx->device, memory, NULL);
    return -1;
}

void
p010_repack_vulkan_cleanup(P010RepackCtx *ctx)
{
    if (!ctx)
        return;

    if (ctx->device != VK_NULL_HANDLE)
        vkDeviceWaitIdle(ctx->device);
    if (ctx->pipeline != VK_NULL_HANDLE)
        vkDestroyPipeline(ctx->device, ctx->pipeline, NULL);
    if (ctx->shader_module != VK_NULL_HANDLE)
        vkDestroyShaderModule(ctx->device, ctx->shader_module, NULL);
    if (ctx->pipeline_layout != VK_NULL_HANDLE)
        vkDestroyPipelineLayout(ctx->device, ctx->pipeline_layout, NULL);
    if (ctx->descriptor_set_layout != VK_NULL_HANDLE)
        vkDestroyDescriptorSetLayout(ctx->device, ctx->descriptor_set_layout, NULL);
    if (ctx->descriptor_pool != VK_NULL_HANDLE)
        vkDestroyDescriptorPool(ctx->device, ctx->descriptor_pool, NULL);
    if (ctx->fence != VK_NULL_HANDLE)
        vkDestroyFence(ctx->device, ctx->fence, NULL);
    if (ctx->command_pool != VK_NULL_HANDLE)
        vkDestroyCommandPool(ctx->device, ctx->command_pool, NULL);
    if (ctx->device != VK_NULL_HANDLE)
        vkDestroyDevice(ctx->device, NULL);
    if (ctx->instance != VK_NULL_HANDLE)
        vkDestroyInstance(ctx->instance, NULL);
    free(ctx);
}

const char *
p010_repack_vulkan_last_error(P010RepackCtx *ctx)
{
    if (!ctx)
        return "P010 repack context is NULL";
    return ctx->last_error;
}
