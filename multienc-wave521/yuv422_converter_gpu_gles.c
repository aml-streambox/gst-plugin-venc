#include "yuv422_converter_gpu_gles.h"

#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <GLES3/gl31.h>
#include <GLES2/gl2ext.h>
#include <wayland-client.h>
#include <wayland-egl.h>

#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <sys/mman.h>
#include <unistd.h>
#include <time.h>

#ifndef DRM_FORMAT_R8
#define DRM_FORMAT_R8 0x20203852
#endif

#ifndef GL_MAP_PERSISTENT_BIT_EXT
#define GL_MAP_PERSISTENT_BIT_EXT 0x0040
#define GL_MAP_COHERENT_BIT_EXT 0x0080
#define GL_DYNAMIC_STORAGE_BIT_EXT 0x0100
#endif

typedef void (GL_APIENTRYP PFNGLBUFFERSTORAGEEXTPROC) (GLenum target, GLsizeiptr size, const void *data, GLbitfield flags);
typedef EGLImageKHR (EGLAPIENTRYP PFNEGLCREATEIMAGEKHRPROC) (EGLDisplay, EGLContext, EGLenum, EGLClientBuffer, const EGLint *);
typedef EGLBoolean (EGLAPIENTRYP PFNEGLDESTROYIMAGEKHRPROC) (EGLDisplay, EGLImageKHR);
typedef void (GL_APIENTRYP PFNGLEGLIMAGETARGETTEXTURE2DOESPROC) (GLenum, GLeglImageOES);

struct GpuCtx {
  struct wl_display *wl_display;
  struct wl_registry *wl_registry;
  struct wl_compositor *wl_compositor;
  struct wl_surface *wl_surface;
  struct wl_egl_window *wl_egl_window;

  EGLDisplay egl_display;
  EGLContext egl_context;
  EGLSurface egl_surface;
  EGLConfig egl_config;

  GLuint program_y;
  GLuint program_uv;
  GLuint program_pack;
  GLuint ssbo_in;
  GLuint ssbo_y;
  GLuint ssbo_uv;

  GLint loc_y_chunks;
  GLint loc_y_pairs;
  GLint loc_y_h;
  GLint loc_uv_chunks;
  GLint loc_uv_pairs;
  GLint loc_uv_half_h;
  GLint loc_pack_tex;

  uint32_t alloc_in_bytes;
  uint32_t alloc_y_bytes;
  uint32_t alloc_uv_bytes;
  int context_bound;
  pthread_t owner_thread;

  int has_buffer_storage;
  PFNGLBUFFERSTORAGEEXTPROC glBufferStorageEXT;
  void *mapped_in;

  PFNEGLCREATEIMAGEKHRPROC eglCreateImageKHR;
  PFNEGLDESTROYIMAGEKHRPROC eglDestroyImageKHR;
  PFNGLEGLIMAGETARGETTEXTURE2DOESPROC glEGLImageTargetTexture2DOES;
  GLuint tex_in;

  char last_error[256];
  int initialized;

  /* Per-instance timing stats (moved from static locals) */
  unsigned frame_counter;
  double sum_upload;
  double sum_y;
  double sum_uv;
  double sum_wait;
  double sum_total;
};

/*
 * Repack P010 to Wave521 MSB format:
 * - Input: LSB-aligned 10-bit in 16-bit word (bits 9-0)
 * - Output: MSB-aligned 10-bit in 16-bit word (bits 15-6), byte-swapped
 */
void repack_p010_to_wave5_msb(uint16_t *y, uint16_t *uv, uint32_t width, uint32_t height)
{
  uint32_t i;
  uint32_t y_count = width * height;
  uint32_t uv_count = (width * height) / 2;

  for (i = 0; i < y_count; i++) {
    uint16_t in = y[i];
    uint16_t v10 = (uint16_t)(in >> 6);
    uint16_t out = (uint16_t)(((v10 & 0x00FFu) << 8) | ((v10 >> 8) & 0x0003u));
    y[i] = out;
  }

  for (i = 0; i < uv_count; i++) {
    uint16_t in = uv[i];
    uint16_t v10 = (uint16_t)(in >> 6);
    uint16_t out = (uint16_t)(((v10 & 0x00FFu) << 8) | ((v10 >> 8) & 0x0003u));
    uv[i] = out;
  }
}

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int check_gl_error(GpuCtx *g, const char *where) {
  GLenum err = glGetError();
  if (err != GL_NO_ERROR) {
    snprintf(g->last_error, sizeof(g->last_error), "%s: GL error 0x%x", where, err);
    return -1;
  }
  return 0;
}

static const char *shader_y =
  "#version 310 es\n"
  "precision highp int;\n"
  "layout(local_size_x = 8, local_size_y = 1) in;\n"
  "layout(std430, binding=0) readonly buffer InB { uint in_data[]; };\n"
  "layout(std430, binding=1) writeonly buffer OutY { uint out_y[]; };\n"
  "const uint PAIRS_PER_ROW = 1920u;\n"
  "const uint TILES_PER_ROW = 60u;\n"
  "shared uint tile_words[40];\n"
  "void main(){\n"
  "  uint lane = gl_LocalInvocationID.x;\n"
  "  uint tile_x = gl_WorkGroupID.x;\n"
  "  uint r = gl_WorkGroupID.y;\n"
  "  uint tile_word_base = (r * TILES_PER_ROW + tile_x) * 40u;\n"
  "  uint lbase = lane * 5u;\n"
  "  tile_words[lbase+0u] = in_data[tile_word_base + lbase + 0u];\n"
  "  tile_words[lbase+1u] = in_data[tile_word_base + lbase + 1u];\n"
  "  tile_words[lbase+2u] = in_data[tile_word_base + lbase + 2u];\n"
  "  tile_words[lbase+3u] = in_data[tile_word_base + lbase + 3u];\n"
  "  tile_words[lbase+4u] = in_data[tile_word_base + lbase + 4u];\n"
  "  barrier();\n"
  "  uint w0=tile_words[lbase+0u], w1=tile_words[lbase+1u], w2=tile_words[lbase+2u], w3=tile_words[lbase+3u], w4=tile_words[lbase+4u];\n"
  "  uint y0_0=(w0>>10u)&0x3FFu; uint y0_1=((w0>>30u)|(w1<<2u))&0x3FFu;\n"
  "  uint y1_0=(w1>>18u)&0x3FFu; uint y1_1=(w2>>6u)&0x3FFu;\n"
  "  uint y2_0=((w2>>26u)|(w3<<6u))&0x3FFu; uint y2_1=(w3>>14u)&0x3FFu;\n"
  "  uint y3_0=(w4>>2u)&0x3FFu; uint y3_1=(w4>>22u)&0x3FFu;\n"
  "  uint p = r * PAIRS_PER_ROW + tile_x*32u + lane*4u;\n"
  // Direct Wave521 format: byte-swap the 10-bit value
  // out = ((val & 0xFF) << 8) | ((val >> 8) & 0x3)
  "  uint y0_0_wave = ((y0_0 & 0xFFu) << 8) | ((y0_0 >> 8) & 0x3u);\n"
  "  uint y0_1_wave = ((y0_1 & 0xFFu) << 8) | ((y0_1 >> 8) & 0x3u);\n"
  "  uint y1_0_wave = ((y1_0 & 0xFFu) << 8) | ((y1_0 >> 8) & 0x3u);\n"
  "  uint y1_1_wave = ((y1_1 & 0xFFu) << 8) | ((y1_1 >> 8) & 0x3u);\n"
  "  uint y2_0_wave = ((y2_0 & 0xFFu) << 8) | ((y2_0 >> 8) & 0x3u);\n"
  "  uint y2_1_wave = ((y2_1 & 0xFFu) << 8) | ((y2_1 >> 8) & 0x3u);\n"
  "  uint y3_0_wave = ((y3_0 & 0xFFu) << 8) | ((y3_0 >> 8) & 0x3u);\n"
  "  uint y3_1_wave = ((y3_1 & 0xFFu) << 8) | ((y3_1 >> 8) & 0x3u);\n"
  "  out_y[p+0u]=y0_0_wave|(y0_1_wave<<16u);\n"
  "  out_y[p+1u]=y1_0_wave|(y1_1_wave<<16u);\n"
  "  out_y[p+2u]=y2_0_wave|(y2_1_wave<<16u);\n"
  "  out_y[p+3u]=y3_0_wave|(y3_1_wave<<16u);\n"
  "}\n";

static const char *shader_uv =
  "#version 310 es\n"
  "precision highp int;\n"
  "layout(local_size_x = 8, local_size_y = 1) in;\n"
  "layout(std430, binding=0) readonly buffer InB { uint in_data[]; };\n"
  "layout(std430, binding=1) writeonly buffer OutUV { uint out_uv[]; };\n"
  "const uint PAIRS_PER_ROW = 1920u;\n"
  "const uint TILES_PER_ROW = 60u;\n"
  "shared uint top_tile[40];\n"
  "shared uint bot_tile[40];\n"
  "void main(){\n"
  "  uint lane = gl_LocalInvocationID.x;\n"
  "  uint tile_x = gl_WorkGroupID.x;\n"
  "  uint ur = gl_WorkGroupID.y;\n"
  "  uint tr = ur*2u; uint br = tr+1u;\n"
  "  uint tile_top_base = (tr * TILES_PER_ROW + tile_x) * 40u;\n"
  "  uint tile_bot_base = (br * TILES_PER_ROW + tile_x) * 40u;\n"
  "  uint lbase = lane * 5u;\n"
  "  top_tile[lbase+0u] = in_data[tile_top_base + lbase + 0u];\n"
  "  top_tile[lbase+1u] = in_data[tile_top_base + lbase + 1u];\n"
  "  top_tile[lbase+2u] = in_data[tile_top_base + lbase + 2u];\n"
  "  top_tile[lbase+3u] = in_data[tile_top_base + lbase + 3u];\n"
  "  top_tile[lbase+4u] = in_data[tile_top_base + lbase + 4u];\n"
  "  bot_tile[lbase+0u] = in_data[tile_bot_base + lbase + 0u];\n"
  "  bot_tile[lbase+1u] = in_data[tile_bot_base + lbase + 1u];\n"
  "  bot_tile[lbase+2u] = in_data[tile_bot_base + lbase + 2u];\n"
  "  bot_tile[lbase+3u] = in_data[tile_bot_base + lbase + 3u];\n"
  "  bot_tile[lbase+4u] = in_data[tile_bot_base + lbase + 4u];\n"
  "  barrier();\n"
  "  uint tw0=top_tile[lbase+0u], tw1=top_tile[lbase+1u], tw2=top_tile[lbase+2u], tw3=top_tile[lbase+3u], tw4=top_tile[lbase+4u];\n"
  "  uint bw0=bot_tile[lbase+0u], bw1=bot_tile[lbase+1u], bw2=bot_tile[lbase+2u], bw3=bot_tile[lbase+3u], bw4=bot_tile[lbase+4u];\n"
  "  uint u0t=tw0&0x3FFu, v0t=(tw0>>20u)&0x3FFu;\n"
  "  uint u1t=(tw1>>8u)&0x3FFu, v1t=((tw1>>28u)|(tw2<<4u))&0x3FFu;\n"
  "  uint u2t=(tw2>>16u)&0x3FFu, v2t=(tw3>>4u)&0x3FFu;\n"
  "  uint u3t=((tw3>>24u)|(tw4<<8u))&0x3FFu, v3t=(tw4>>12u)&0x3FFu;\n"
  "  uint u0b=bw0&0x3FFu, v0b=(bw0>>20u)&0x3FFu;\n"
  "  uint u1b=(bw1>>8u)&0x3FFu, v1b=((bw1>>28u)|(bw2<<4u))&0x3FFu;\n"
  "  uint u2b=(bw2>>16u)&0x3FFu, v2b=(bw3>>4u)&0x3FFu;\n"
  "  uint u3b=((bw3>>24u)|(bw4<<8u))&0x3FFu, v3b=(bw4>>12u)&0x3FFu;\n"
  "  uint p = ur * PAIRS_PER_ROW + tile_x*32u + lane*4u;\n"
  // UV averaging and Wave521 byte-swap
  "  uint u0=((u0t+u0b+1u)>>1u), v0=((v0t+v0b+1u)>>1u);\n"
  "  uint u1=((u1t+u1b+1u)>>1u), v1=((v1t+v1b+1u)>>1u);\n"
  "  uint u2=((u2t+u2b+1u)>>1u), v2=((v2t+v2b+1u)>>1u);\n"
  "  uint u3=((u3t+u3b+1u)>>1u), v3=((v3t+v3b+1u)>>1u);\n"
  "  uint u0_wave = ((u0 & 0xFFu) << 8) | ((u0 >> 8) & 0x3u);\n"
  "  uint v0_wave = ((v0 & 0xFFu) << 8) | ((v0 >> 8) & 0x3u);\n"
  "  uint u1_wave = ((u1 & 0xFFu) << 8) | ((u1 >> 8) & 0x3u);\n"
  "  uint v1_wave = ((v1 & 0xFFu) << 8) | ((v1 >> 8) & 0x3u);\n"
  "  uint u2_wave = ((u2 & 0xFFu) << 8) | ((u2 >> 8) & 0x3u);\n"
  "  uint v2_wave = ((v2 & 0xFFu) << 8) | ((v2 >> 8) & 0x3u);\n"
  "  uint u3_wave = ((u3 & 0xFFu) << 8) | ((u3 >> 8) & 0x3u);\n"
  "  uint v3_wave = ((v3 & 0xFFu) << 8) | ((v3 >> 8) & 0x3u);\n"
  "  out_uv[p+0u]=u0_wave|(v0_wave<<16u);\n"
  "  out_uv[p+1u]=u1_wave|(v1_wave<<16u);\n"
  "  out_uv[p+2u]=u2_wave|(v2_wave<<16u);\n"
  "  out_uv[p+3u]=u3_wave|(v3_wave<<16u);\n"
  "}\n";

static const char *shader_pack =
  "#version 310 es\n"
  "precision highp int;\n"
  "precision highp usampler2D;\n"
  "layout(local_size_x = 64, local_size_y = 1) in;\n"
  "layout(binding=0) uniform highp usampler2D in_tex;\n"
  "layout(std430, binding=0) writeonly buffer InWords { uint in_data[]; };\n"
  "const uint WORDS_PER_ROW = 2400u;\n"
  "const uint HEIGHT = 2160u;\n"
  "void main(){\n"
  "  uint wx = gl_GlobalInvocationID.x;\n"
  "  uint y = gl_GlobalInvocationID.y;\n"
  "  if (wx >= WORDS_PER_ROW || y >= HEIGHT) return;\n"
  "  uint x = wx * 4u;\n"
  "  uint b0 = texelFetch(in_tex, ivec2(int(x+0u), int(y)), 0).r;\n"
  "  uint b1 = texelFetch(in_tex, ivec2(int(x+1u), int(y)), 0).r;\n"
  "  uint b2 = texelFetch(in_tex, ivec2(int(x+2u), int(y)), 0).r;\n"
  "  uint b3 = texelFetch(in_tex, ivec2(int(x+3u), int(y)), 0).r;\n"
  "  in_data[y * WORDS_PER_ROW + wx] = b0 | (b1<<8u) | (b2<<16u) | (b3<<24u);\n"
  "}\n";

static void registry_global(void *data, struct wl_registry *reg, uint32_t id, const char *iface, uint32_t version) {
  (void)version;
  GpuCtx *ctx = (GpuCtx *)data;
  if (strcmp(iface, wl_compositor_interface.name) == 0)
    ctx->wl_compositor = wl_registry_bind(reg, id, &wl_compositor_interface, 1);
}

static void registry_remove(void *data, struct wl_registry *reg, uint32_t id) {
  (void)data; (void)reg; (void)id;
}

static const struct wl_registry_listener reg_listener = { registry_global, registry_remove };

static GLuint compile_cs(GpuCtx *g, const char *src) {
  GLuint s = glCreateShader(GL_COMPUTE_SHADER);
  glShaderSource(s, 1, &src, NULL);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[512] = {0};
    glGetShaderInfoLog(s, sizeof(log), NULL, log);
    snprintf(g->last_error, sizeof(g->last_error), "compute shader compile failed: %s", log);
    glDeleteShader(s);
    return 0;
  }
  return s;
}

static GLuint link_program(GpuCtx *g, GLuint shader) {
  GLuint p = glCreateProgram();
  glAttachShader(p, shader);
  glLinkProgram(p);
  glDeleteShader(shader);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[512] = {0};
    glGetProgramInfoLog(p, sizeof(log), NULL, log);
    snprintf(g->last_error, sizeof(g->last_error), "program link failed: %s", log);
    glDeleteProgram(p);
    return 0;
  }
  return p;
}

GpuCtx *yuv422_gpu_gles_init(void) {
  GpuCtx *g = calloc(1, sizeof(GpuCtx));
  if (!g) {
    fprintf(stderr, "[GLES-ERR] Failed to allocate GpuCtx\n");
    return NULL;
  }

  if (!getenv("WAYLAND_DISPLAY") || !*getenv("WAYLAND_DISPLAY"))
    setenv("WAYLAND_DISPLAY", "wayland-0", 1);

  g->wl_display = wl_display_connect(NULL);
  if (!g->wl_display) {
    snprintf(g->last_error, sizeof(g->last_error), "wl_display_connect failed");
    free(g);
    return NULL;
  }

  g->wl_registry = wl_display_get_registry(g->wl_display);
  wl_registry_add_listener(g->wl_registry, &reg_listener, g);
  wl_display_roundtrip(g->wl_display);
  if (!g->wl_compositor) {
    snprintf(g->last_error, sizeof(g->last_error), "no wayland compositor");
    wl_registry_destroy(g->wl_registry);
    wl_display_disconnect(g->wl_display);
    free(g);
    return NULL;
  }

  g->egl_display = eglGetDisplay((EGLNativeDisplayType)g->wl_display);
  if (g->egl_display == EGL_NO_DISPLAY) {
    snprintf(g->last_error, sizeof(g->last_error), "eglGetDisplay failed");
    free(g);
    return NULL;
  }

  EGLint maj = 0, min = 0;
  if (!eglInitialize(g->egl_display, &maj, &min)) {
    snprintf(g->last_error, sizeof(g->last_error), "eglInitialize failed");
    free(g);
    return NULL;
  }

  if (!eglBindAPI(EGL_OPENGL_ES_API)) {
    snprintf(g->last_error, sizeof(g->last_error), "eglBindAPI failed");
    free(g);
    return NULL;
  }

  EGLint ncfg = 0;
  EGLConfig cfgs[64];
  eglGetConfigs(g->egl_display, cfgs, 64, &ncfg);
  for (EGLint i = 0; i < ncfg; i++) {
    EGLint st = 0, rt = 0;
    eglGetConfigAttrib(g->egl_display, cfgs[i], EGL_SURFACE_TYPE, &st);
    eglGetConfigAttrib(g->egl_display, cfgs[i], EGL_RENDERABLE_TYPE, &rt);
    if ((st & EGL_WINDOW_BIT) && (rt & EGL_OPENGL_ES3_BIT)) {
      g->egl_config = cfgs[i];
      break;
    }
  }

  EGLint ctx_attr[] = { EGL_CONTEXT_CLIENT_VERSION, 3, EGL_NONE };
  g->egl_context = eglCreateContext(g->egl_display, g->egl_config, EGL_NO_CONTEXT, ctx_attr);
  if (g->egl_context == EGL_NO_CONTEXT) {
    snprintf(g->last_error, sizeof(g->last_error), "eglCreateContext failed");
    free(g);
    return NULL;
  }

  g->wl_surface = wl_compositor_create_surface(g->wl_compositor);
  g->wl_egl_window = wl_egl_window_create(g->wl_surface, 1, 1);
  g->egl_surface = eglCreateWindowSurface(g->egl_display, g->egl_config, (EGLNativeWindowType)g->wl_egl_window, NULL);
  if (g->egl_surface == EGL_NO_SURFACE) {
    snprintf(g->last_error, sizeof(g->last_error), "eglCreateWindowSurface failed");
    free(g);
    return NULL;
  }

  if (!eglMakeCurrent(g->egl_display, g->egl_surface, g->egl_surface, g->egl_context)) {
    snprintf(g->last_error, sizeof(g->last_error), "eglMakeCurrent failed");
    free(g);
    return NULL;
  }

  g->program_y = link_program(g, compile_cs(g, shader_y));
  if (!g->program_y) {
    free(g);
    return NULL;
  }
  g->program_uv = link_program(g, compile_cs(g, shader_uv));
  if (!g->program_uv) {
    free(g);
    return NULL;
  }
  g->program_pack = link_program(g, compile_cs(g, shader_pack));
  if (!g->program_pack) {
    free(g);
    return NULL;
  }
  g->loc_pack_tex = glGetUniformLocation(g->program_pack, "in_tex");

  g->loc_y_chunks = glGetUniformLocation(g->program_y, "u_chunks_per_row");
  g->loc_y_pairs = glGetUniformLocation(g->program_y, "u_pairs_per_row");
  g->loc_y_h = glGetUniformLocation(g->program_y, "u_h");
  g->loc_uv_chunks = glGetUniformLocation(g->program_uv, "u_chunks_per_row");
  g->loc_uv_pairs = glGetUniformLocation(g->program_uv, "u_pairs_per_row");
  g->loc_uv_half_h = glGetUniformLocation(g->program_uv, "u_half_h");

  glGenBuffers(1, &g->ssbo_in);
  glGenBuffers(1, &g->ssbo_y);
  glGenBuffers(1, &g->ssbo_uv);
  glGenTextures(1, &g->tex_in);

  const char *exts = (const char *)glGetString(GL_EXTENSIONS);
  g->has_buffer_storage = (exts && strstr(exts, "GL_EXT_buffer_storage"));
  if (g->has_buffer_storage) {
    g->glBufferStorageEXT = (PFNGLBUFFERSTORAGEEXTPROC)eglGetProcAddress("glBufferStorageEXT");
    if (!g->glBufferStorageEXT)
      g->has_buffer_storage = 0;
  }

  g->eglCreateImageKHR = (PFNEGLCREATEIMAGEKHRPROC)eglGetProcAddress("eglCreateImageKHR");
  g->eglDestroyImageKHR = (PFNEGLDESTROYIMAGEKHRPROC)eglGetProcAddress("eglDestroyImageKHR");
  g->glEGLImageTargetTexture2DOES = (PFNGLEGLIMAGETARGETTEXTURE2DOESPROC)eglGetProcAddress("glEGLImageTargetTexture2DOES");

  /* Release current context from init thread.
   * Streaming thread will bind it in convert(). */
  eglMakeCurrent(g->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);

  g->initialized = 1;
  return g;
}

int yuv422_gpu_gles_convert_p010(GpuCtx *g, const ConversionParams *params) {
  if (!g || !g->initialized) {
    if (g) snprintf(g->last_error, sizeof(g->last_error), "gpu not initialized");
    return -1;
  }

  if (!params || !params->input_data) {
    snprintf(g->last_error, sizeof(g->last_error), "invalid params");
    return -1;
  }

  int compute_only = (params->output_y == NULL || params->output_uv == NULL);
  double t0 = now_ms();

  pthread_t self_tid = pthread_self();
  if (!g->context_bound || !pthread_equal(g->owner_thread, self_tid)) {
    if (!eglMakeCurrent(g->egl_display, g->egl_surface, g->egl_surface, g->egl_context)) {
      snprintf(g->last_error, sizeof(g->last_error), "eglMakeCurrent failed in convert");
      return -1;
    }
    g->context_bound = 1;
    g->owner_thread = self_tid;
  }

  const uint32_t w = params->width;
  const uint32_t h = params->height;
  const uint32_t pairs = w / 2;
  const uint32_t half_h = h / 2;

  if (w != 3840 || h != 2160) {
    snprintf(g->last_error, sizeof(g->last_error), "fixed-kernel requires 3840x2160");
    return -1;
  }

  const uint32_t in_bytes = w * h * 5 / 2;
  const uint32_t y_bytes = w * h * 2;
  const uint32_t uv_bytes = pairs * half_h * 4;

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->ssbo_in);
  if (g->alloc_in_bytes < in_bytes) {
    if (g->has_buffer_storage && g->glBufferStorageEXT) {
      if (g->mapped_in) {
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
        g->mapped_in = NULL;
      }
      g->glBufferStorageEXT(GL_SHADER_STORAGE_BUFFER, in_bytes, NULL,
          GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT_EXT | GL_MAP_COHERENT_BIT_EXT | GL_DYNAMIC_STORAGE_BIT_EXT);
      g->mapped_in = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, in_bytes,
          GL_MAP_WRITE_BIT | GL_MAP_PERSISTENT_BIT_EXT | GL_MAP_COHERENT_BIT_EXT);
      if (!g->mapped_in) {
        g->has_buffer_storage = 0;
        glBufferData(GL_SHADER_STORAGE_BUFFER, in_bytes, NULL, GL_DYNAMIC_DRAW);
      }
    } else {
      glBufferData(GL_SHADER_STORAGE_BUFFER, in_bytes, NULL, GL_DYNAMIC_DRAW);
    }
    g->alloc_in_bytes = in_bytes;
  }
  if (g->has_buffer_storage && g->mapped_in) {
    memcpy(g->mapped_in, params->input_data, in_bytes);
  } else {
    glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, in_bytes, params->input_data);
  }
  if (check_gl_error(g, "upload input")) return -1;
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, g->ssbo_in);
  double t1 = now_ms();

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->ssbo_y);
  if (g->alloc_y_bytes < y_bytes) {
    glBufferData(GL_SHADER_STORAGE_BUFFER, y_bytes, NULL, GL_DYNAMIC_DRAW);
    g->alloc_y_bytes = y_bytes;
  }
  if (check_gl_error(g, "alloc y")) return -1;
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, g->ssbo_y);

  glUseProgram(g->program_y);
  glDispatchCompute(60, 2160, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  if (check_gl_error(g, "dispatch y")) return -1;
  double t2 = now_ms();

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->ssbo_uv);
  if (g->alloc_uv_bytes < uv_bytes) {
    glBufferData(GL_SHADER_STORAGE_BUFFER, uv_bytes, NULL, GL_DYNAMIC_DRAW);
    g->alloc_uv_bytes = uv_bytes;
  }
  if (check_gl_error(g, "alloc uv")) return -1;
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, g->ssbo_uv);

  glUseProgram(g->program_uv);
  glDispatchCompute(60, 1080, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);
  if (check_gl_error(g, "dispatch uv")) return -1;
  double t3 = now_ms();

  GLsync fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  glFlush();
  glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 5000000000ULL);
  glDeleteSync(fence);
  if (check_gl_error(g, "wait fence")) return -1;
  double t4 = now_ms();

  /* Per-instance timing stats (no longer static globals) */
  g->frame_counter++;
  g->sum_upload += (t1 - t0);
  g->sum_y += (t2 - t1);
  g->sum_uv += (t3 - t2);
  g->sum_wait += (t4 - t3);
  g->sum_total += (t4 - t0);

  if ((g->frame_counter % 30u) == 0u) {
    fprintf(stderr,
        "[v10conv-gpu] avg30 compute-only stages: upload=%.3fms y=%.3fms uv=%.3fms wait=%.3fms total=%.3fms\n",
        g->sum_upload / 30.0, g->sum_y / 30.0, g->sum_uv / 30.0, g->sum_wait / 30.0, g->sum_total / 30.0);
    g->sum_upload = g->sum_y = g->sum_uv = g->sum_wait = g->sum_total = 0.0;
  }

  if (compute_only) {
    return 0;
  }

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->ssbo_y);
  void *ym = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, y_bytes, GL_MAP_READ_BIT);
  if (!ym) {
    snprintf(g->last_error, sizeof(g->last_error), "map Y buffer failed");
    return -1;
  }
  memcpy(params->output_y, ym, y_bytes);
  glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->ssbo_uv);
  void *uvm = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, uv_bytes, GL_MAP_READ_BIT);
  if (!uvm) {
    snprintf(g->last_error, sizeof(g->last_error), "map UV buffer failed");
    return -1;
  }
  memcpy(params->output_uv, uvm, uv_bytes);
  glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);

  if (check_gl_error(g, "readback")) return -1;

  return 0;
}

int yuv422_gpu_gles_convert_p010_dmabuf(GpuCtx *g, int in_fd, int out_fd, uint32_t width, uint32_t height)
{
  if (!g) return -1;
  if (in_fd < 0 || out_fd < 0) {
    snprintf(g->last_error, sizeof(g->last_error), "invalid dmabuf fd");
    return -1;
  }

  size_t in_bytes = width * height * 5 / 2;
  size_t out_bytes = width * height * 2 + (width / 2) * (height / 2) * 4;
  size_t y_bytes = width * height * 2;

  void *in_map = mmap(NULL, in_bytes, PROT_READ, MAP_SHARED, in_fd, 0);
  if (in_map == MAP_FAILED) {
    snprintf(g->last_error, sizeof(g->last_error), "mmap input dmabuf failed");
    return -1;
  }

  void *out_map = mmap(NULL, out_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, out_fd, 0);
  if (out_map == MAP_FAILED) {
    munmap(in_map, in_bytes);
    snprintf(g->last_error, sizeof(g->last_error), "mmap output dmabuf failed");
    return -1;
  }

  ConversionParams p;
  memset(&p, 0, sizeof(p));
  p.width = width;
  p.height = height;
  p.input_data = (const uint8_t *)in_map;
  p.output_y = (uint16_t *)out_map;
  p.output_uv = (uint16_t *)((uint8_t *)out_map + y_bytes);
  p.output_format = OUTPUT_YUV420_P010;

  int ret = yuv422_gpu_gles_convert_p010(g, &p);

  munmap(out_map, out_bytes);
  munmap(in_map, in_bytes);
  return ret;
}

int yuv422_gpu_gles_compute_p010_dmabuf(GpuCtx *g, int in_fd, uint32_t width, uint32_t height)
{
  if (!g || !g->initialized) return -1;
  if (in_fd < 0 || width != 3840 || height != 2160) {
    snprintf(g->last_error, sizeof(g->last_error), "invalid dmabuf input for compute path");
    return -1;
  }
  if (!g->eglCreateImageKHR || !g->eglDestroyImageKHR || !g->glEGLImageTargetTexture2DOES) {
    snprintf(g->last_error, sizeof(g->last_error), "EGL image import extensions missing");
    return -1;
  }

  pthread_t self_tid = pthread_self();
  if (!g->context_bound || !pthread_equal(g->owner_thread, self_tid)) {
    if (!eglMakeCurrent(g->egl_display, g->egl_surface, g->egl_surface, g->egl_context)) {
      snprintf(g->last_error, sizeof(g->last_error), "eglMakeCurrent failed in compute dmabuf");
      return -1;
    }
    g->context_bound = 1;
    g->owner_thread = self_tid;
  }

  EGLint attrs[] = {
    EGL_WIDTH, 9600,
    EGL_HEIGHT, 2160,
    EGL_LINUX_DRM_FOURCC_EXT, DRM_FORMAT_R8,
    EGL_DMA_BUF_PLANE0_FD_EXT, in_fd,
    EGL_DMA_BUF_PLANE0_OFFSET_EXT, 0,
    EGL_DMA_BUF_PLANE0_PITCH_EXT, 9600,
    EGL_NONE
  };

  EGLImageKHR image = g->eglCreateImageKHR(g->egl_display, EGL_NO_CONTEXT, EGL_LINUX_DMA_BUF_EXT, (EGLClientBuffer)NULL, attrs);
  if (image == EGL_NO_IMAGE_KHR) {
    snprintf(g->last_error, sizeof(g->last_error), "eglCreateImageKHR(dmabuf) failed");
    return -1;
  }

  uint32_t in_bytes = width * height * 5 / 2;
  uint32_t y_bytes = width * height * 2;
  uint32_t uv_bytes = (width / 2) * (height / 2) * 4;

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->ssbo_in);
  if (g->alloc_in_bytes < in_bytes) { glBufferData(GL_SHADER_STORAGE_BUFFER, in_bytes, NULL, GL_DYNAMIC_DRAW); g->alloc_in_bytes = in_bytes; }
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, g->ssbo_in);

  glBindTexture(GL_TEXTURE_2D, g->tex_in);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  g->glEGLImageTargetTexture2DOES(GL_TEXTURE_2D, (GLeglImageOES)image);

  glUseProgram(g->program_pack);
  glActiveTexture(GL_TEXTURE0);
  glBindTexture(GL_TEXTURE_2D, g->tex_in);
  glUniform1i(g->loc_pack_tex, 0);
  glDispatchCompute((2400 + 63) / 64, 2160, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT | GL_TEXTURE_FETCH_BARRIER_BIT);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->ssbo_y);
  if (g->alloc_y_bytes < y_bytes) { glBufferData(GL_SHADER_STORAGE_BUFFER, y_bytes, NULL, GL_DYNAMIC_DRAW); g->alloc_y_bytes = y_bytes; }
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, g->ssbo_y);
  glUseProgram(g->program_y);
  glDispatchCompute(60, 2160, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

  glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->ssbo_uv);
  if (g->alloc_uv_bytes < uv_bytes) { glBufferData(GL_SHADER_STORAGE_BUFFER, uv_bytes, NULL, GL_DYNAMIC_DRAW); g->alloc_uv_bytes = uv_bytes; }
  glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, g->ssbo_uv);
  glUseProgram(g->program_uv);
  glDispatchCompute(60, 1080, 1);
  glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT);

  GLsync sync_fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
  glFlush();
  glClientWaitSync(sync_fence, GL_SYNC_FLUSH_COMMANDS_BIT, 5000000000ULL);
  glDeleteSync(sync_fence);

  g->eglDestroyImageKHR(g->egl_display, image);
  return 0;
}

void yuv422_gpu_gles_cleanup(GpuCtx *g) {
  if (!g)
    return;
  if (!g->initialized) {
    free(g);
    return;
  }
  if (g->mapped_in) {
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, g->ssbo_in);
    glUnmapBuffer(GL_SHADER_STORAGE_BUFFER);
    g->mapped_in = NULL;
  }
  if (g->program_pack) glDeleteProgram(g->program_pack);
  if (g->program_y) glDeleteProgram(g->program_y);
  if (g->program_uv) glDeleteProgram(g->program_uv);
  if (g->tex_in) glDeleteTextures(1, &g->tex_in);
  if (g->ssbo_in) glDeleteBuffers(1, &g->ssbo_in);
  if (g->ssbo_y) glDeleteBuffers(1, &g->ssbo_y);
  if (g->ssbo_uv) glDeleteBuffers(1, &g->ssbo_uv);
  if (g->egl_display != EGL_NO_DISPLAY) {
    eglMakeCurrent(g->egl_display, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
    if (g->egl_surface != EGL_NO_SURFACE) eglDestroySurface(g->egl_display, g->egl_surface);
    if (g->egl_context != EGL_NO_CONTEXT) eglDestroyContext(g->egl_display, g->egl_context);
    eglTerminate(g->egl_display);
  }
  if (g->wl_egl_window) wl_egl_window_destroy(g->wl_egl_window);
  if (g->wl_surface) wl_surface_destroy(g->wl_surface);
  if (g->wl_compositor) wl_compositor_destroy(g->wl_compositor);
  if (g->wl_registry) wl_registry_destroy(g->wl_registry);
  if (g->wl_display) wl_display_disconnect(g->wl_display);
  free(g);
}

const char *yuv422_gpu_gles_last_error(GpuCtx *g) {
  if (!g) return "NULL context";
  return g->last_error;
}
