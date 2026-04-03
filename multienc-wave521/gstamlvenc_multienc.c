/*
 * Copyright (C) 2014-2019 Amlogic, Inc. All rights reserved.
 *
 * All information contained herein is Amlogic confidential.
 */

/**
 * SECTION:element-amlvenc
 *
 * FIXME:Describe amlvenc here.
 *
 * <refsect2>
 * <title>Example launch line</title>
 * |[
 * gst-launch -v -m fakesrc ! amlvenc ! fakesink silent=TRUE
 * ]|
 * </refsect2>
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

//#include <gmodule.h>
//#include <gst/allocators/gstamlionallocator.h>
#include <gst/gstdrmbufferpool.h>
#include <gst/allocators/gstdmabuf.h>
#include <gst/gst.h>
#include <gst/pbutils/pbutils.h>
#include <gst/video/gstvideometa.h>
#include <gst/video/gstvideopool.h>
#include <gst/video/gstvideosink.h>
#include <gst/video/video.h>
#include <gst/allocators/gstdmabuf.h>
#include <linux/dma-heap.h>
#include <linux/dma-buf.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <signal.h>
#include <glib-unix.h>

#include "gstamlvenc_multienc.h"
#include "imgproc.h"

#include "gstamlionallocator.h"
#include "yuv422_converter_gpu_gles.h"
#include "yuv422_converter_vulkan.h"

GST_DEBUG_CATEGORY_STATIC (gst_amlvenc_debug);
#define GST_CAT_DEFAULT gst_amlvenc_debug

static gboolean
gst_amlvenc_add_v_chroma_format (GstAmlVEnc *encoder, GstStructure * s)
{
  GValue fmts = G_VALUE_INIT;
  GValue fmt = G_VALUE_INIT;
  gboolean ret = FALSE;

  g_value_init (&fmts, GST_TYPE_LIST);
  g_value_init (&fmt, G_TYPE_STRING);

  g_value_set_string (&fmt, "NV12");
  gst_value_list_append_value (&fmts, &fmt);
  g_value_set_string (&fmt, "NV21");
  gst_value_list_append_value (&fmts, &fmt);
  g_value_set_string (&fmt, "I420");
  gst_value_list_append_value (&fmts, &fmt);
  g_value_set_string (&fmt, "YV12");
  gst_value_list_append_value (&fmts, &fmt);
  g_value_set_string (&fmt, "RGB");
  gst_value_list_append_value (&fmts, &fmt);
  g_value_set_string (&fmt, "BGR");
  gst_value_list_append_value (&fmts, &fmt);
  g_value_set_string (&fmt, "P010_10LE");
  gst_value_list_append_value (&fmts, &fmt);
  g_value_set_string (&fmt, "ENCODED");
  gst_value_list_append_value (&fmts, &fmt);

  if (gst_value_list_get_size (&fmts) != 0) {
    gst_structure_take_value (s, "format", &fmts);
    ret = TRUE;
  } else {
    g_value_unset (&fmts);
  }

  g_value_unset (&fmt);

  return ret;
}

#define PROP_IDR_PERIOD_DEFAULT 30
#define PROP_FRAMERATE_DEFAULT 30
#define PROP_BITRATE_DEFAULT 2000
#define PROP_BITRATE_MAX G_MAXINT  /* Removed 12Mbps limit - encoder can handle 300Mbps+ */
#define PROP_MIN_BUFFERS_DEFAULT 2
#define PROP_MAX_BUFFERS_DEFAULT 6
#define PROP_ENCODER_BUFFER_SIZE_DEFAULT 2048
#define PROP_ENCODER_BUFFER_SIZE_MIN 1024
#define PROP_ENCODER_BUFFER_SIZE_MAX 4096

#define PROP_ROI_ID_DEFAULT 0
#define PROP_ROI_ENABLED_DEFAULT TRUE
#define PROP_ROI_WIDTH_DEFAULT 0.00
#define PROP_ROI_HEIGHT_DEFAULT 0.00
#define PROP_ROI_X_DEFAULT 0.00
#define PROP_ROI_Y_DEFAULT 0.00
#define PROP_ROI_QUALITY_DEFAULT 51

#define PROP_INTERNAL_BIT_DEPTH_DEFAULT 8
#define PROP_GOP_PATTERN_DEFAULT 0
#define PROP_RC_MODE_DEFAULT 0
#define PROP_LOSSLESS_ENABLE_DEFAULT FALSE
#define PROP_V10CONV_BACKEND_DEFAULT 0  /* 0=vulkan, 1=gles */

#define PROP_QP_B_DEFAULT 0
#define PROP_MIN_QP_B_DEFAULT 8
#define PROP_MAX_QP_B_DEFAULT 51

#define DRMBP_EXTRA_BUF_SIZE_FOR_DISPLAY 1
#define DRMBP_LIMIT_MAX_BUFSIZE_TO_BUFSIZE 1

static int
gst_amlvenc_alloc_dma_heap_fd (gsize size)
{
  const char *heaps[] = {
    "/dev/dma_heap/heap-codecmm",
    "/dev/dma_heap/heap-cached-codecmm",
    "/dev/dma_heap/heap-gfx",
    "/dev/dma_heap/system-uncached",
    "/dev/dma_heap/system",
    NULL
  };
  int i;

  for (i = 0; heaps[i] != NULL; i++) {
    int heapfd = open (heaps[i], O_RDWR | O_CLOEXEC);
    struct dma_heap_allocation_data alloc_data;

    if (heapfd < 0)
      continue;

    memset (&alloc_data, 0, sizeof (alloc_data));
    alloc_data.len = size;
    alloc_data.fd_flags = O_RDWR | O_CLOEXEC;

    if (ioctl (heapfd, DMA_HEAP_IOCTL_ALLOC, &alloc_data) == 0) {
      close (heapfd);
      return alloc_data.fd;
    }

    close (heapfd);
  }

  return -1;
}

static void
gst_amlvenc_clear_v10conv_buffers (GstAmlVEnc * encoder)
{
  if (encoder->v10conv.output_y.memory) {
    gst_memory_unref (encoder->v10conv.output_y.memory);
    encoder->v10conv.output_y.memory = NULL;
  }
  if (encoder->v10conv.output_uv.memory) {
    gst_memory_unref (encoder->v10conv.output_uv.memory);
    encoder->v10conv.output_uv.memory = NULL;
  }
  /* Close dup'd fds we own */
  if (encoder->v10conv.output_y.fd_dup >= 0) {
    close (encoder->v10conv.output_y.fd_dup);
    encoder->v10conv.output_y.fd_dup = -1;
  }
  if (encoder->v10conv.output_uv.fd_dup >= 0) {
    close (encoder->v10conv.output_uv.fd_dup);
    encoder->v10conv.output_uv.fd_dup = -1;
  }
  encoder->v10conv.output_y.fd = -1;
  encoder->v10conv.output_uv.fd = -1;
  
  /* Clean up double-buffered outputs */
  for (int i = 0; i < 2; i++) {
    if (encoder->v10conv.output_buf[i].memory) {
      gst_memory_unref (encoder->v10conv.output_buf[i].memory);
      encoder->v10conv.output_buf[i].memory = NULL;
    }
    encoder->v10conv.output_buf[i].fd = -1;
  }
  encoder->v10conv.write_idx = 0;
  encoder->v10conv.pipeline_primed = FALSE;

  /* Clean up Vulkan GPU resources if initialized */
  if (encoder->v10conv.vulkan_ctx) {
    yuv422_vulkan_cleanup(encoder->v10conv.vulkan_ctx);
    encoder->v10conv.vulkan_ctx = NULL;
  }

  /* Clean up GLES GPU resources if initialized */
  if (encoder->v10conv.gles_ctx) {
    yuv422_gpu_gles_cleanup(encoder->v10conv.gles_ctx);
    encoder->v10conv.gles_ctx = NULL;
  }
}

static gboolean
gst_amlvenc_prepare_v10conv_buffers (GstAmlVEnc * encoder, const GstVideoInfo * info)
{
  gsize out_size = info->width * info->height * 3;

  /* Already allocated? */
  if (encoder->v10conv.output_buf[0].memory)
    return TRUE;

  if (!encoder->v10conv.allocator)
    encoder->v10conv.allocator = gst_dmabuf_allocator_new ();

  if (!encoder->v10conv.allocator)
    return FALSE;

  /* Allocate double-buffered output dmabufs for pipelined GPU+encoder */
  for (int i = 0; i < 2; i++) {
    int out_fd = gst_amlvenc_alloc_dma_heap_fd (out_size);
    if (out_fd < 0)
      goto fail;

    encoder->v10conv.output_buf[i].memory =
        gst_dmabuf_allocator_alloc (encoder->v10conv.allocator,
        out_fd, out_size);
    if (!encoder->v10conv.output_buf[i].memory) {
      close (out_fd);
      goto fail;
    }

    encoder->v10conv.output_buf[i].fd =
        gst_dmabuf_memory_get_fd (encoder->v10conv.output_buf[i].memory);
  }

  /* Also set output_y for backward compat (GLES path, P010 stats logging) */
  encoder->v10conv.output_y.memory = encoder->v10conv.output_buf[0].memory;
  gst_memory_ref (encoder->v10conv.output_y.memory);
  encoder->v10conv.output_y.fd = encoder->v10conv.output_buf[0].fd;
  encoder->v10conv.output_uv.fd = -1;

  encoder->v10conv.write_idx = 0;
  encoder->v10conv.pipeline_primed = FALSE;
  return TRUE;

fail:
  gst_amlvenc_clear_v10conv_buffers (encoder);
  return FALSE;
}

enum
{
  PROP_0,
  PROP_GOP,
  PROP_FRAMERATE,
  PROP_BITRATE,
  PROP_MIN_BUFFERS,
  PROP_MAX_BUFFERS,
  PROP_ENCODER_BUFSIZE,
  PROP_ROI_ID,
  PROP_ROI_ENABLED,
  PROP_ROI_WIDTH,
  PROP_ROI_HEIGHT,
  PROP_ROI_X,
  PROP_ROI_Y,
  PROP_ROI_QUALITY,
  PROP_INTERNAL_BIT_DEPTH,
  PROP_GOP_PATTERN,
  PROP_RC_MODE,
  PROP_LOSSLESS_ENABLE,
  PROP_QP_B,
  PROP_MIN_QP_B,
  PROP_MAX_QP_B,
  PROD_ENABLE_DMALLOCATOR,
  PROP_V10CONV_BACKEND
};

struct aml_roi_location {
  gfloat left;
  gfloat top;
  gfloat width;
  gfloat height;
};

struct RoiParamInfo {
  struct listnode list;
  gint id;
  gint quality;
  struct aml_roi_location location;
};

#define COMMON_SRC_PADS \
        "framerate = (fraction) [0/1, MAX], " \
        "width = (int) [ 1, MAX ], " "height = (int) [ 1, MAX ], " \
        "stream-format = (string) { byte-stream }, " \
        "alignment = (string) au, " \
        "profile = (string) { high-4:4:4, high-4:2:2, high-10, high, main," \
        " baseline, constrained-baseline, high-4:4:4-intra, high-4:2:2-intra," \
        " high-10-intra }"

static GstStaticPadTemplate src_factory = GST_STATIC_PAD_TEMPLATE ("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS ("video/x-h264, "
        COMMON_SRC_PADS "; "
        "video/x-h265, "
        COMMON_SRC_PADS)
    );

static void gst_amlvenc_finalize (GObject * object);
static gboolean gst_amlvenc_start (GstVideoEncoder * encoder);
static gboolean gst_amlvenc_stop (GstVideoEncoder * encoder);
static gboolean gst_amlvenc_flush (GstVideoEncoder * encoder);
static int gst_amlvenc_alloc_dma_heap_fd (gsize size);
static gboolean gst_amlvenc_prepare_v10conv_buffers (GstAmlVEnc * encoder,
    const GstVideoInfo * info);
static void gst_amlvenc_clear_v10conv_buffers (GstAmlVEnc * encoder);

static gboolean gst_amlvenc_init_encoder (GstAmlVEnc * encoder);
static gboolean gst_amlvenc_set_roi(GstAmlVEnc * encoder);
static void gst_amlvenc_fill_roi_buffer(guchar* buffer, gint buffer_w, gint buffer_h,
    struct RoiParamInfo *param_info, gint vframe_w, gint vframe_h, gint block_size);
static void gst_amlvenc_close_encoder (GstAmlVEnc * encoder);

static GstFlowReturn gst_amlvenc_finish (GstVideoEncoder * encoder);
static GstFlowReturn gst_amlvenc_handle_frame (GstVideoEncoder * encoder,
    GstVideoCodecFrame * frame);
static GstFlowReturn gst_amlvenc_encode_frame (GstAmlVEnc * encoder,
    GstVideoCodecFrame * frame);
static gboolean gst_amlvenc_set_format (GstVideoEncoder * video_enc,
    GstVideoCodecState * state);

static gboolean gst_amlvenc_propose_allocation (GstVideoEncoder * encoder,
    GstQuery * query);

static void gst_amlvenc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec);
static void gst_amlvenc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec);

#define gst_amlvenc_parent_class parent_class
G_DEFINE_TYPE_WITH_CODE (GstAmlVEnc, gst_amlvenc, GST_TYPE_VIDEO_ENCODER,
    G_IMPLEMENT_INTERFACE (GST_TYPE_PRESET, NULL));

static vl_img_format_t
img_format_convert (GstVideoFormat vfmt) {
  vl_img_format_t fmt;
  switch (vfmt) {
  case GST_VIDEO_FORMAT_NV12:
    fmt = IMG_FMT_NV12;
    break;
  case GST_VIDEO_FORMAT_NV21:
    fmt = IMG_FMT_NV21;
    break;
  case GST_VIDEO_FORMAT_I420:
  case GST_VIDEO_FORMAT_YV12:
    fmt = IMG_FMT_YUV420P;
    break;
  case GST_VIDEO_FORMAT_RGB:
  case GST_VIDEO_FORMAT_BGR:
    // use ge2d for internal conversation
    fmt = IMG_FMT_NV12;
    break;
  case GST_VIDEO_FORMAT_P010_10LE:
    fmt = IMG_FMT_P010;
    break;
  default:
    fmt = IMG_FMT_NONE;
    break;
  }
  return fmt;
}

/* allowed input caps depending on whether libv was built for 8 or 10 bits */
static GstCaps *
gst_amlvenc_sink_getcaps (GstVideoEncoder * enc, GstCaps * filter)
{
  GstCaps *supported_incaps;
  GstCaps *allowed;
  GstCaps *filter_caps, *fcaps;
  gint i, j;

  supported_incaps =
      gst_pad_get_pad_template_caps (GST_VIDEO_ENCODER_SINK_PAD (enc));

  /* Allow downstream to specify width/height/framerate/PAR constraints
   * and forward them upstream for video converters to handle
   */
  allowed = gst_pad_get_allowed_caps (enc->srcpad);

  if (!allowed || gst_caps_is_empty (allowed) || gst_caps_is_any (allowed)) {
    fcaps = supported_incaps;
    goto done;
  }

  GST_LOG_OBJECT (enc, "template caps %" GST_PTR_FORMAT, supported_incaps);
  GST_LOG_OBJECT (enc, "allowed caps %" GST_PTR_FORMAT, allowed);

  filter_caps = gst_caps_new_empty ();

  for (i = 0; i < gst_caps_get_size (supported_incaps); i++) {
    GQuark q_name =
        gst_structure_get_name_id (gst_caps_get_structure (supported_incaps,
            i));

    for (j = 0; j < gst_caps_get_size (allowed); j++) {
      const GstStructure *allowed_s = gst_caps_get_structure (allowed, j);
      const GValue *val;
      GstStructure *s;
      const gchar* allowed_mime_name = gst_structure_get_name (allowed_s);
      GstAmlVEnc *venc = GST_AMLVENC (enc);

      if (!g_strcmp0 (allowed_mime_name, "video/x-h265"))
      {
        venc->codec.id = CODEC_ID_H265;
      } else if (!g_strcmp0 (allowed_mime_name, "video/x-h264")) {
        venc->codec.id = CODEC_ID_H264;
      }

      s = gst_structure_new_id_empty (q_name);
      if ((val = gst_structure_get_value (allowed_s, "width")))
        gst_structure_set_value (s, "width", val);
      if ((val = gst_structure_get_value (allowed_s, "height")))
        gst_structure_set_value (s, "height", val);
      if ((val = gst_structure_get_value (allowed_s, "framerate")))
        gst_structure_set_value (s, "framerate", val);
      if ((val = gst_structure_get_value (allowed_s, "pixel-aspect-ratio")))
        gst_structure_set_value (s, "pixel-aspect-ratio", val);

      gst_amlvenc_add_v_chroma_format (venc, s);

      filter_caps = gst_caps_merge_structure (filter_caps, s);
    }
  }

  fcaps = gst_caps_intersect (filter_caps, supported_incaps);
  gst_caps_unref (filter_caps);
  gst_caps_unref (supported_incaps);

  if (filter) {
    GST_LOG_OBJECT (enc, "intersecting with %" GST_PTR_FORMAT, filter);
    filter_caps = gst_caps_intersect (fcaps, filter);
    gst_caps_unref (fcaps);
    fcaps = filter_caps;
  }

done:
  gst_caps_replace (&allowed, NULL);

  GST_LOG_OBJECT (enc, "proxy caps %" GST_PTR_FORMAT, fcaps);

  return fcaps;
}

static gboolean
gst_amlvenc_sink_query (GstVideoEncoder * enc, GstQuery * query)
{
  GstPad *pad = GST_VIDEO_ENCODER_SINK_PAD (enc);
  gboolean ret = FALSE;

  GST_DEBUG ("Received %s query on sinkpad, %" GST_PTR_FORMAT,
      GST_QUERY_TYPE_NAME (query), query);

  switch (GST_QUERY_TYPE (query)) {
    case GST_QUERY_ACCEPT_CAPS:{
      GstCaps *acceptable, *caps;

      acceptable = gst_pad_get_pad_template_caps (pad);

      gst_query_parse_accept_caps (query, &caps);

      gst_query_set_accept_caps_result (query,
          gst_caps_is_subset (caps, acceptable));
      gst_caps_unref (acceptable);
      ret = TRUE;
    }
      break;
    default:
      ret = GST_VIDEO_ENCODER_CLASS (parent_class)->sink_query (enc, query);
      break;
  }

  return ret;
}

static void cleanup_roi_param_list (GstAmlVEnc *encoder) {
  struct listnode *pos, *q;
  if (!list_empty(&encoder->roi.param_info)) {
    list_for_each_safe(pos, q, &encoder->roi.param_info) {
      struct RoiParamInfo *param_info =
        list_entry (pos, struct RoiParamInfo, list);
      list_remove (pos);

      g_free(param_info);
    }
  }
}

static struct RoiParamInfo *retrieve_roi_param_info(GstAmlVEnc *encoder, gint id) {
  GstAmlVEnc *self = GST_AMLVENC (encoder);
  struct RoiParamInfo *ret = NULL;
  if (!list_empty (&self->roi.param_info)) {
    struct listnode *pos;
    list_for_each (pos, &self->roi.param_info) {
      struct RoiParamInfo *param_info =
        list_entry (pos, struct RoiParamInfo, list);
      if (param_info->id == id) {
        ret = param_info;
      }
    }
  }
  if (ret == NULL) {
    ret = g_new(struct RoiParamInfo, 1);
    list_init (&ret->list);
    list_add_tail(&self->roi.param_info, &ret->list);
    ret->id = id;
    ret->location.left = PROP_ROI_X_DEFAULT;
    ret->location.top = PROP_ROI_Y_DEFAULT;
    ret->location.width = PROP_ROI_WIDTH_DEFAULT;
    ret->location.height = PROP_ROI_HEIGHT_DEFAULT;
    ret->quality = PROP_ROI_QUALITY_DEFAULT;
  }
  return ret;
}

static void
gst_amlvenc_class_init (GstAmlVEncClass * klass)
{
  GObjectClass *gobject_class;
  GstElementClass *element_class;
  GstVideoEncoderClass *gstencoder_class;
  GstPadTemplate *sink_templ;
  GstCaps *supported_sinkcaps;

  gobject_class = G_OBJECT_CLASS (klass);
  element_class = GST_ELEMENT_CLASS (klass);
  gstencoder_class = GST_VIDEO_ENCODER_CLASS (klass);

  gobject_class->set_property = gst_amlvenc_set_property;
  gobject_class->get_property = gst_amlvenc_get_property;
  gobject_class->finalize = gst_amlvenc_finalize;

  gstencoder_class->set_format = GST_DEBUG_FUNCPTR (gst_amlvenc_set_format);
  gstencoder_class->handle_frame =
      GST_DEBUG_FUNCPTR (gst_amlvenc_handle_frame);
  gstencoder_class->start = GST_DEBUG_FUNCPTR (gst_amlvenc_start);
  gstencoder_class->stop = GST_DEBUG_FUNCPTR (gst_amlvenc_stop);
  gstencoder_class->flush = GST_DEBUG_FUNCPTR (gst_amlvenc_flush);
  gstencoder_class->finish = GST_DEBUG_FUNCPTR (gst_amlvenc_finish);
  gstencoder_class->getcaps = GST_DEBUG_FUNCPTR (gst_amlvenc_sink_getcaps);
  gstencoder_class->propose_allocation =
      GST_DEBUG_FUNCPTR (gst_amlvenc_propose_allocation);
  gstencoder_class->sink_query = GST_DEBUG_FUNCPTR (gst_amlvenc_sink_query);

  g_object_class_install_property (gobject_class, PROP_GOP,
      g_param_spec_int ("gop", "GOP", "IDR frame refresh interval",
          -1, 1000, PROP_IDR_PERIOD_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_FRAMERATE,
      g_param_spec_int ("framerate", "Framerate", "framerate(fps)",
          0, 240, PROP_FRAMERATE_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_BITRATE,
      g_param_spec_int ("bitrate", "Bitrate", "bitrate(kbps)",
          0, PROP_BITRATE_MAX, PROP_BITRATE_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MIN_BUFFERS,
      g_param_spec_int ("min-buffers", "Min-Buffers", "min number of input buffer",
          0, 2, PROP_MIN_BUFFERS_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MAX_BUFFERS,
      g_param_spec_int ("max-buffers", "Max-Buffers", "max number of input buffer",
          3, 10, PROP_MAX_BUFFERS_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_ENCODER_BUFSIZE,
      g_param_spec_int ("encoder-buffer-size", "Encoder-Buffer-Size", "Encoder Buffer Size(KBytes)",
          PROP_ENCODER_BUFFER_SIZE_MIN, PROP_ENCODER_BUFFER_SIZE_MAX, PROP_ENCODER_BUFFER_SIZE_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_ROI_ENABLED,
      g_param_spec_boolean ("roi-enabled", "roi-enabled", "Enable/Disable the roi function",
          PROP_ROI_ENABLED_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property(gobject_class, PROP_ROI_ID,
      g_param_spec_int("roi-id", "roi-id", "Current roi operation id",
          0, G_MAXINT32, PROP_ROI_ID_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_ROI_WIDTH,
      g_param_spec_float("roi-width", "roi-width", "Relative width of the roi rectangle",
          0.00, G_MAXFLOAT, PROP_ROI_WIDTH_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_ROI_HEIGHT,
      g_param_spec_float("roi-height", "roi-height", "Relative height of the roi rectangle",
          0.00, G_MAXFLOAT, PROP_ROI_HEIGHT_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_ROI_X,
      g_param_spec_float("roi-x", "roi-x", "Relative horizontal start position of the roi rectangle",
          0.00, G_MAXFLOAT, PROP_ROI_X_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_ROI_Y,
      g_param_spec_float("roi-y", "roi-y", "Relative vertical start position of the roi rectangle",
          0.00, G_MAXFLOAT, PROP_ROI_Y_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_ROI_QUALITY,
      g_param_spec_int("roi-quality", "roi-quality", "Quality of roi area",
          0, 51, PROP_ROI_QUALITY_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_INTERNAL_BIT_DEPTH,
      g_param_spec_int("internal-bit-depth", "internal-bit-depth", "Encoder internal bit depth (8 or 10)",
          8, 10, PROP_INTERNAL_BIT_DEPTH_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_GOP_PATTERN,
      g_param_spec_int("gop-pattern", "gop-pattern", "GOP structure pattern (0=IP, 1=IBBBP, 2=IBPBP, 3=ALL_I)",
          0, 4, PROP_GOP_PATTERN_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_RC_MODE,
      g_param_spec_int("rc-mode", "rc-mode", "Rate control mode (0=VBR, 1=CBR, 2=CQP)",
          0, 2, PROP_RC_MODE_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_LOSSLESS_ENABLE,
      g_param_spec_boolean("lossless-enable", "lossless-enable", "Enable lossless encoding (HEVC only - disables rate control, NR, ROI)",
          PROP_LOSSLESS_ENABLE_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_QP_B,
      g_param_spec_int("qp-b", "qp-b", "Base QP for B-frames (0=auto)",
          0, 51, PROP_QP_B_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MIN_QP_B,
      g_param_spec_int("min-qp-b", "min-qp-b", "Minimum QP for B-frames",
          0, 51, PROP_MIN_QP_B_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_MAX_QP_B,
      g_param_spec_int("max-qp-b", "max-qp-b", "Maximum QP for B-frames",
          0, 51, PROP_MAX_QP_B_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROD_ENABLE_DMALLOCATOR,
      g_param_spec_boolean ("enable-dmallocator", "enable-dmallocator", "Enable/Disable dmallocator",
          FALSE,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  g_object_class_install_property (gobject_class, PROP_V10CONV_BACKEND,
      g_param_spec_int ("v10conv-backend", "v10conv-backend", "V10 conversion backend (0=vulkan, 1=gles)",
          0, 1, PROP_V10CONV_BACKEND_DEFAULT,
          G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS));

  gst_element_class_set_static_metadata (element_class,
    "Amlogic h264/h265 Multi-Encoder",
    "Codec/Encoder/Video",
    "Amlogic h264/h265 Multi-Encoder Plugin",
    "Jemy Zhang <jun.zhang@amlogic.com>");

  supported_sinkcaps = gst_caps_new_simple ("video/x-raw",
      "framerate", GST_TYPE_FRACTION_RANGE, 0, 1, G_MAXINT, 1,
      "width", GST_TYPE_INT_RANGE, 16, G_MAXINT,
      "height", GST_TYPE_INT_RANGE, 16, G_MAXINT, NULL);

  gst_amlvenc_add_v_chroma_format (NULL, gst_caps_get_structure (supported_sinkcaps, 0));

  sink_templ = gst_pad_template_new ("sink",
      GST_PAD_SINK, GST_PAD_ALWAYS, supported_sinkcaps);

  gst_caps_unref (supported_sinkcaps);

  gst_element_class_add_pad_template (element_class, sink_templ);
  gst_element_class_add_static_pad_template (element_class, &src_factory);
}

/* initialize the new element
 * instantiate pads and add them to element
 * set functions
 * initialize structure
 */
static void
gst_amlvenc_init (GstAmlVEnc * encoder)
{
  encoder->gop = PROP_IDR_PERIOD_DEFAULT;
  encoder->framerate = PROP_FRAMERATE_DEFAULT;
  encoder->bitrate = PROP_BITRATE_DEFAULT;
  encoder->max_buffers = PROP_MAX_BUFFERS_DEFAULT;
  encoder->min_buffers = PROP_MIN_BUFFERS_DEFAULT;
  encoder->encoder_bufsize = PROP_ENCODER_BUFFER_SIZE_DEFAULT * 1024;
  encoder->codec.id = CODEC_ID_NONE;
  encoder->internal_bit_depth = PROP_INTERNAL_BIT_DEPTH_DEFAULT;
  encoder->gop_pattern = PROP_GOP_PATTERN_DEFAULT;
  encoder->rc_mode = PROP_RC_MODE_DEFAULT;
  encoder->lossless_enable = PROP_LOSSLESS_ENABLE_DEFAULT;
  encoder->v10conv_backend = PROP_V10CONV_BACKEND_DEFAULT;
  
  /* B-frame QP properties */
  encoder->qp_b = PROP_QP_B_DEFAULT;
  encoder->min_qp_b = PROP_MIN_QP_B_DEFAULT;
  encoder->max_qp_b = PROP_MAX_QP_B_DEFAULT;
  
  /* B-frame reordering state */
  encoder->bframe_enabled = FALSE;
  encoder->gop_size = 1;
  encoder->frame_counter = 0;
  encoder->reorder_count = 0;
  encoder->frame_duration = GST_CLOCK_TIME_NONE;
  memset(encoder->reorder_queue, 0, sizeof(encoder->reorder_queue));

  list_init(&encoder->roi.param_info);
  encoder->roi.srcid = 0;
  encoder->roi.block_size = 16;    // H264
  encoder->roi.enabled = PROP_ROI_ENABLED_DEFAULT;
  encoder->roi.id = PROP_ROI_ID_DEFAULT;
  encoder->roi.buffer_info.data = NULL;
  encoder->fd[0] = -1;
  encoder->fd[1] = -1;
  encoder->fd[2] = -1;

  encoder->u4_first_pts_index = 0;
  encoder->b_enable_dmallocator = TRUE;

  encoder->stopping = FALSE;
  encoder->sigint_source_id = 0;
  g_mutex_init (&encoder->encoder_lock);
}

static void
gst_amlvenc_finalize (GObject * object)
{
  GstAmlVEnc *encoder = GST_AMLVENC (object);
  g_mutex_clear (&encoder->encoder_lock);
  G_OBJECT_CLASS (parent_class)->finalize (object);
}


/* ---- SIGUSR1-driven EOS injection ---- */

static gboolean
gst_amlvenc_sigusr1_handler (gpointer user_data)
{
  GstAmlVEnc *encoder = GST_AMLVENC (user_data);

  GST_WARNING_OBJECT (encoder,
      "SIGUSR1 caught by amlvenc — injecting EOS for clean encoder shutdown");

  /* Set stopping flag so encode paths bail out immediately */
  encoder->stopping = TRUE;

  /* Inject EOS on the encoder element.
   * This arrives on the sink pad, causing the GstVideoEncoder base class
   * to call our finish() vfunc for a clean flush, then propagates
   * downstream so the whole pipeline shuts down gracefully.
   * This bypasses any blocked upstream element (e.g. v4l2src in poll/DQBUF). */
  gst_element_send_event (GST_ELEMENT (encoder), gst_event_new_eos ());

  /* Invalidate the source ID — this callback won't run again */
  encoder->sigint_source_id = 0;

  return G_SOURCE_REMOVE;
}

static gboolean
gst_amlvenc_start (GstVideoEncoder * encoder)
{
  GstAmlVEnc *venc = GST_AMLVENC (encoder);

  /* Reset stopping flag for fresh pipeline run */
  venc->stopping = FALSE;

  venc->dmabuf_alloc = gst_amlion_allocator_obtain();

  if (venc->codec.buf == NULL) {
    venc->codec.buf = g_new (guchar, venc->encoder_bufsize);
  }
  venc->imgproc.input.memory = NULL;
  venc->imgproc.output.memory = NULL;
  venc->v10conv.enabled = FALSE;
  venc->v10conv.allocator = NULL;
  venc->v10conv.output_y.memory = NULL;
  venc->v10conv.output_uv.memory = NULL;
  venc->v10conv.output_y.fd = -1;
  venc->v10conv.output_uv.fd = -1;
  venc->v10conv.output_y.fd_dup = -1;
  venc->v10conv.output_uv.fd_dup = -1;
  /* make sure that we have enough time for first DTS,
     this is probably overkill for most streams */
  gst_video_encoder_set_min_pts (encoder, GST_MSECOND * 30);

  /* Install SIGUSR1 handler so the pipeline manager can trigger a clean EOS
   * shutdown that bypasses any blocked upstream element (e.g. v4l2src).
   * SIGUSR1 is used instead of SIGINT to avoid interfering with GStreamer's
   * own Ctrl+C / SIGINT handling in gst-launch-1.0 -e.
   * g_unix_signal_add uses an internal pipe — safe from signal context. */
  venc->sigint_source_id = g_unix_signal_add (SIGUSR1,
      gst_amlvenc_sigusr1_handler, venc);
  GST_WARNING_OBJECT (venc, "start: installed SIGUSR1 handler (source_id=%u)",
      venc->sigint_source_id);

  return TRUE;
}

static gboolean
gst_amlvenc_stop (GstVideoEncoder * encoder)
{
  GstAmlVEnc *venc = GST_AMLVENC (encoder);

  /* Signal all blocking paths to bail out */
  venc->stopping = TRUE;
  GST_WARNING_OBJECT (venc, "stop: setting stopping flag, closing encoder");

  /* Remove SIGUSR1 handler to avoid firing after encoder is torn down */
  if (venc->sigint_source_id) {
    g_source_remove (venc->sigint_source_id);
    venc->sigint_source_id = 0;
  }

  gst_amlvenc_close_encoder (venc);

  if (venc->roi.srcid)
    g_source_remove (venc->roi.srcid);

  if (venc->input_state) {
    gst_video_codec_state_unref (venc->input_state);
    venc->input_state = NULL;
  }

  if (venc->codec.buf) {
    g_free((gpointer)venc->codec.buf);
    venc->codec.buf = NULL;
  }

  if (venc->dmabuf_alloc) {
    gst_object_unref(venc->dmabuf_alloc);
    venc->dmabuf_alloc = NULL;
  }

  if (venc->imgproc.input.memory) {
    gst_memory_unref(venc->imgproc.input.memory);
    venc->imgproc.input.memory = NULL;
  }

  if (venc->imgproc.output.memory) {
    gst_memory_unref(venc->imgproc.output.memory);
    venc->imgproc.output.memory = NULL;
  }

  gst_amlvenc_clear_v10conv_buffers (venc);
  if (venc->v10conv.allocator) {
    gst_object_unref (venc->v10conv.allocator);
    venc->v10conv.allocator = NULL;
  }

  if (venc->roi.buffer_info.data) {
    g_free((gpointer)venc->roi.buffer_info.data);
    venc->roi.buffer_info.data = NULL;
  }

  cleanup_roi_param_list(venc);

  return TRUE;
}


static gboolean
gst_amlvenc_flush (GstVideoEncoder * encoder)
{
  GstAmlVEnc *venc = GST_AMLVENC (encoder);

  gst_amlvenc_init_encoder (venc);

  return TRUE;
}

/*
 * gst_amlvenc_init_encoder
 * @encoder:  Encoder which should be initialized.
 *
 * Initialize v encoder.
 *
 */
static gboolean
gst_amlvenc_init_encoder (GstAmlVEnc * encoder)
{
  GstVideoInfo *info;
  GstVideoInfo p010_info;

  if (!encoder->input_state) {
    GST_DEBUG_OBJECT (encoder, "Have no input state yet");
    return FALSE;
  }

  info = &encoder->input_state->info;

  /* make sure that the encoder is closed */
  gst_amlvenc_close_encoder (encoder);

  GST_OBJECT_LOCK (encoder);


  GST_OBJECT_UNLOCK (encoder);

  vl_encode_info_t encode_info;
  memset (&encode_info, 0, sizeof(vl_encode_info_t));

  encode_info.width = info->width;
  encode_info.height = info->height;
  encode_info.frame_rate = encoder->framerate;
  encode_info.bit_rate = encoder->bitrate * 1000;
  encode_info.gop = encoder->gop;
  encode_info.img_format = img_format_convert(GST_VIDEO_INFO_FORMAT(info));
  encode_info.prepend_spspps_to_idr_frames = TRUE;
  encode_info.enc_feature_opts |= 0x1;  // enable roi function
  encode_info.internal_bit_depth = encoder->internal_bit_depth;
  encode_info.gop_pattern = encoder->gop_pattern;
  switch (encoder->gop_pattern) {
    case 1: /* IBBBP */
      encode_info.enc_feature_opts |= (3 << 2);
      break;
    case 4: /* ALL_I */
      encode_info.enc_feature_opts |= (1 << 2);
      break;
    default:
      break;
  }
  encode_info.rc_mode = encoder->rc_mode;
  encode_info.lossless_enable = encoder->lossless_enable;
  /* FIX: For lossless encoding, increase bitstream buffer size if not explicitly set */
  if (encoder->lossless_enable) {
    /* Default buffer size may be too small for lossless (3-5x larger output) */
    if (encoder->encoder_bufsize < 4096 * 1024) { // if less than 4MB
      guint new_bufsize = 4096 * 1024; // 4MB minimum for lossless
      GST_WARNING_OBJECT (encoder, "lossless mode: increasing encoder_bufsize from %dKB to %dKB",
          encoder->encoder_bufsize / 1024, new_bufsize / 1024);
      encoder->encoder_bufsize = new_bufsize;
    }
  }
  encode_info.bitstream_buf_sz_kb = encoder->encoder_bufsize / 1024;

  if (encoder->codec.buf == NULL) {
    encoder->codec.buf = g_malloc0 (encoder->encoder_bufsize);
  } else {
    encoder->codec.buf = g_realloc (encoder->codec.buf, encoder->encoder_bufsize);
  }

  encoder->v10conv.enabled = (GST_VIDEO_INFO_FORMAT (info) == GST_VIDEO_FORMAT_ENCODED);
  encoder->output_counter = 0;
  encoder->bframe_base_pts = GST_CLOCK_TIME_NONE;
  encoder->codec_header_sent = FALSE;
  if (encoder->codec_header) {
    g_free (encoder->codec_header);
    encoder->codec_header = NULL;
    encoder->codec_header_size = 0;
  }

  /* VUI color signaling: read colorimetry from incoming caps and convert to
   * ITU-T H.273 integer codes for the H.264/H.265 SPS VUI.  This covers all
   * input paths — standard video formats carrying colorimetry in caps as well
   * as the v10conv (ENCODED) path where we fall back to BT.2020+PQ. */
  {
    GstVideoColorimetry *cinfo = &info->colorimetry;
    guint cp = gst_video_color_primaries_to_iso (cinfo->primaries);
    guint tc = gst_video_transfer_function_to_iso (cinfo->transfer);
    guint mc = gst_video_color_matrix_to_iso (cinfo->matrix);

    /* v10conv path: caps carry ENCODED format so GstVideoInfo won't have
     * useful colorimetry — default to BT.2020 + PQ (HDR10). */
    if (encoder->v10conv.enabled && cp == 0 && tc == 0 && mc == 0) {
      cp = 9;   /* BT.2020 */
      tc = 16;  /* SMPTE ST 2084 (PQ) */
      mc = 9;   /* BT.2020 non-constant luminance */
      GST_INFO_OBJECT (encoder,
          "v10conv path: no colorimetry in caps, defaulting to BT.2020 + PQ");
    }

    if (cp != 0 || tc != 0 || mc != 0) {
      encode_info.vui_parameters_present_flag = 1;
      encode_info.video_signal_type_present_flag = 1;
      encode_info.video_full_range_flag =
          (cinfo->range == GST_VIDEO_COLOR_RANGE_0_255) ? 1 : 0;
      encode_info.colour_description_present_flag = 1;
      encode_info.colour_primaries = (uint8_t) cp;
      encode_info.transfer_characteristics = (uint8_t) tc;
      encode_info.matrix_coefficients = (uint8_t) mc;
      GST_INFO_OBJECT (encoder,
          "VUI signaling: primaries=%u transfer=%u matrix=%u range=%d",
          cp, tc, mc, encode_info.video_full_range_flag);
    } else {
      GST_DEBUG_OBJECT (encoder,
          "No colorimetry in caps, VUI will not be signaled");
    }
  }

  if (encoder->v10conv.enabled) {
      /*
       * Build a virtual P010_10LE video info for proper encoder initialization.
       * The source may have ENCODED format with different semantics, but after
       * internal GPU conversion we produce standard P010_10LE.
       * Use source dimensions and framerate for the virtual info.
       */
      gst_video_info_init (&p010_info);
      gst_video_info_set_format (&p010_info, GST_VIDEO_FORMAT_P010_10LE, info->width, info->height);
      /* Preserve source framerate (e.g., 60fps), don't use encoder property default (30fps) */
      p010_info.fps_n = info->fps_n;
      p010_info.fps_d = info->fps_d;
      /* Use virtual P010 info for encoder parameters */
      info = &p010_info;
      encode_info.img_format = IMG_FMT_P010;
      /* Use source framerate for encoding, not the encoder property default */
      if (info->fps_n > 0) {
        encode_info.frame_rate = info->fps_n / info->fps_d;
        /* Update encoder framerate property to match source for logging/consistency */
        encoder->framerate = encode_info.frame_rate;
      }
      if (!encoder->v10conv.gles_ctx) {
        encoder->v10conv.gles_ctx = yuv422_gpu_gles_init ();
        if (!encoder->v10conv.gles_ctx) {
          GST_ELEMENT_ERROR (encoder, STREAM, ENCODE,
              ("Can not initialize internal v10 converter."), (NULL));
          return FALSE;
        }
      }
      if (!gst_amlvenc_prepare_v10conv_buffers (encoder, info)) {
        GST_ELEMENT_ERROR (encoder, STREAM, ENCODE,
            ("Can not allocate internal v10 conversion buffers."), (NULL));
        return FALSE;
      }
  } else {
      gst_amlvenc_clear_v10conv_buffers (encoder);
  }

  GST_WARNING_OBJECT (encoder,
      "init encoder request: codec=%d format=%s %dx%d fps=%d bitrate=%d gop=%d depth=%d roi=%d min=%d max=%d dmalloc=%d gop_pattern=%d rc_mode=%d lossless=%d v10conv=%d bs_buf=%dKB",
      encoder->codec.id,
      gst_video_format_to_string (GST_VIDEO_INFO_FORMAT (info)),
      info->width, info->height,
      encoder->framerate,
      encoder->bitrate * 1000,
      encoder->gop,
      encoder->internal_bit_depth,
      encoder->roi.enabled ? 1 : 0,
      encoder->min_buffers,
      encoder->max_buffers,
      encoder->b_enable_dmallocator ? 1 : 0,
      encoder->gop_pattern,
      encoder->rc_mode,
      encoder->lossless_enable ? 1 : 0,
      encoder->v10conv.enabled ? 1 : 0,
      (int)encode_info.bitstream_buf_sz_kb);

  qp_param_t qp_tbl;
  memset(&qp_tbl, 0, sizeof(qp_param_t));

  qp_tbl.qp_min = 0;
  qp_tbl.qp_max = 51;
  qp_tbl.qp_I_base = 30;
  qp_tbl.qp_I_min = 0;
  qp_tbl.qp_I_max = 51;
  qp_tbl.qp_P_base = 30;
  qp_tbl.qp_P_min = 0;
  qp_tbl.qp_P_max = 51;
  qp_tbl.qp_B_base = encoder->qp_b;
  qp_tbl.qp_B_min = encoder->min_qp_b;
  qp_tbl.qp_B_max = encoder->max_qp_b;

  encoder->codec.handle = vl_multi_encoder_init(encoder->codec.id, encode_info, &qp_tbl);

  if (encoder->codec.handle == 0) {
    GST_WARNING_OBJECT (encoder,
        "vl_multi_encoder_init failed: codec=%d format=%s %dx%d fps=%d bitrate=%d depth=%d roi=%d qp_b=%d min_qp_b=%d max_qp_b=%d",
        encoder->codec.id,
        gst_video_format_to_string (GST_VIDEO_INFO_FORMAT (info)),
        info->width, info->height,
        encoder->framerate,
        encoder->bitrate * 1000,
        encoder->internal_bit_depth,
        encoder->roi.enabled ? 1 : 0,
        encoder->qp_b,
        encoder->min_qp_b,
        encoder->max_qp_b);
    GST_ELEMENT_ERROR (encoder, STREAM, ENCODE,
        ("Can not initialize v encoder."), (NULL));
    return FALSE;
  }

  if (!gst_amlvenc_set_roi (encoder)) {
    return FALSE;
  }
  
  /* Initialize B-frame reordering state based on GOP pattern */
  encoder->frame_counter = 0;
  encoder->reorder_count = 0;
  encoder->bframe_enabled = FALSE;
  encoder->gop_size = 1;
  
  /* Calculate frame duration for timestamp handling */
  if (info->fps_n > 0 && info->fps_d > 0) {
    encoder->frame_duration = gst_util_uint64_scale (GST_SECOND, info->fps_d, info->fps_n);
  } else {
    encoder->frame_duration = GST_SECOND / 30; /* default 30fps */
  }
  
  /* Check if B-frames are enabled based on GOP pattern */
  switch (encoder->gop_pattern) {
    case 1: /* IBBBP */
      encoder->bframe_enabled = TRUE;
      encoder->gop_size = 4;
      GST_INFO_OBJECT (encoder, "B-frames enabled: IBBBP pattern, GOP size %d", encoder->gop_size);
      break;
    case 2: /* IBPBP */
      encoder->bframe_enabled = TRUE;
      encoder->gop_size = 2;
      GST_INFO_OBJECT (encoder, "B-frames enabled: IBPBP pattern, GOP size %d", encoder->gop_size);
      break;
    case 3: /* IBBB */
      encoder->bframe_enabled = TRUE;
      encoder->gop_size = 4;
      GST_INFO_OBJECT (encoder, "B-frames enabled: IBBB pattern, GOP size %d", encoder->gop_size);
      break;
    default:
      encoder->bframe_enabled = FALSE;
      encoder->gop_size = 1;
      GST_INFO_OBJECT (encoder, "B-frames disabled: GOP pattern %d", encoder->gop_pattern);
      break;
  }
  
  /* Clear reorder queue */
  memset(encoder->reorder_queue, 0, sizeof(encoder->reorder_queue));

  if (GST_VIDEO_INFO_FORMAT(info) == GST_VIDEO_FORMAT_RGB ||
      GST_VIDEO_INFO_FORMAT(info) == GST_VIDEO_FORMAT_BGR) {
    encoder->imgproc.handle = imgproc_init();
    if (encoder->imgproc.handle == NULL) {
      GST_ELEMENT_ERROR (encoder, STREAM, ENCODE,
          ("Can not initialize imgproc."), (NULL));
      return FALSE;
    }
    encoder->imgproc.outbuf_size = (info->width * info->height * 3) / 2;
  }

  if (encoder->codec.handle != 0 && encoder->codec.id == CODEC_ID_H265 && encoder->bframe_enabled) {
    guint header_alloc = MAX (encoder->encoder_bufsize, 256u) * 1024;
    guint8 *header = g_malloc0 (header_alloc);
    guint header_size = header_alloc;
    encoding_metadata_t header_meta;

    header_meta = vl_multi_encoder_generate_header (encoder->codec.handle,
        header, &header_size);
    if (header_meta.is_valid && header_size > 0) {
      encoder->codec_header = header;
      encoder->codec_header_size = header_size;
      GST_INFO_OBJECT (encoder, "cached codec header size=%u for first output prepend",
          encoder->codec_header_size);
    } else {
      GST_WARNING_OBJECT (encoder,
          "failed to cache codec header for first output prepend (valid=%d size=%u err=%d)",
          header_meta.is_valid, header_size, header_meta.err_cod);
      g_free (header);
    }
  }

  return TRUE;
}


/*
 * gst_amlvenc_set_roi
 * @encoder:  update encoder roi value.
 *
 * Set roi value
 */
static gboolean
gst_amlvenc_set_roi(GstAmlVEnc * encoder)
{
  GstVideoInfo *info;

  if (!encoder->input_state) {
    GST_DEBUG_OBJECT (encoder, "Have no input state yet");
    return FALSE;
  }

  info = &encoder->input_state->info;
  gint vframe_w = info->width;
  gint vframe_h = info->height;

  gint buffer_w = encoder->roi.buffer_info.width;
  gint buffer_h = encoder->roi.buffer_info.height;

  if (encoder->roi.enabled) {
    struct listnode *pos = NULL;
    struct RoiParamInfo *param_info = NULL;
    list_for_each(pos, &encoder->roi.param_info) {
      param_info = list_entry(pos, struct RoiParamInfo, list);
      GST_DEBUG("roi-id:%d, roi-left:%.6f, roi-top:%.6f, roi-width:%.6f, roi-height:%.6f, roi-quality:%d\n",
              param_info->id,
              param_info->location.left,
              param_info->location.top,
              param_info->location.width,
              param_info->location.height,
              param_info->quality);

      gst_amlvenc_fill_roi_buffer(
              encoder->roi.buffer_info.data,
              buffer_w, buffer_h,
              param_info,
              vframe_w,
              vframe_h,
              encoder->roi.block_size
            );
    }
  }

  gint ret;
  if (encoder->codec.handle) {
    if ((ret = vl_video_encoder_update_qp_hint(
            encoder->codec.handle,
            encoder->roi.buffer_info.data,
            buffer_w * buffer_h)) != 0) {
      GST_DEBUG_OBJECT (encoder, "update roi value failed, ret:%d\n", ret);
      return FALSE;
    }
  }
  return TRUE;
}

static void
gst_amlvenc_fill_roi_buffer(guchar* buffer, gint buffer_w, gint buffer_h,
    struct RoiParamInfo *param_info, gint vframe_w, gint vframe_h, gint block_size) {
  if (buffer == NULL || param_info == NULL) return;

  gint left = param_info->location.left * vframe_w;
  gint top = param_info->location.top * vframe_h;
  gint width = param_info->location.width * vframe_w;
  gint height = param_info->location.height * vframe_h;

  gint right = left + width;
  gint bottom = top + height;

  gint limit = block_size / 2;

  gint start_row = top / block_size;
  gint start_col = left / block_size;
  if ((left % block_size) > limit) start_col += 1;
  if ((top % block_size) > limit) start_row += 1;

  gint stop_row = bottom / block_size;
  gint stop_col = right / block_size;
  if ((right % block_size) >= limit) stop_col += 1;
  if ((bottom % block_size) >= limit) stop_row += 1;

  if (start_row <= stop_row && start_col <= stop_col) {
    for (int i_row = start_row; i_row < stop_row; i_row++) {
      for (int j_col = start_col; j_col < stop_col; j_col++) {
        buffer[i_row * buffer_w + j_col] = param_info->quality;
      }
    }
  }
}

static gboolean
idle_set_roi(GstAmlVEnc * self) {
  if (self != NULL) {
    gst_amlvenc_set_roi (self);
  }

  self->roi.srcid = 0;
  return G_SOURCE_REMOVE;
}


/* gst_amlvenc_close_encoder
 * @encoder:  Encoder which should close.
 *
 * Close v encoder.
 */
static void
gst_amlvenc_close_encoder (GstAmlVEnc * encoder)
{
  g_mutex_lock (&encoder->encoder_lock);
  if (encoder->codec.handle != 0) {
    vl_multi_encoder_destroy(encoder->codec.handle);
    encoder->codec.handle = 0;
  }
  g_mutex_unlock (&encoder->encoder_lock);
  if (encoder->imgproc.handle) {
    imgproc_deinit(encoder->imgproc.handle);
    encoder->imgproc.handle = NULL;
  }
  if (encoder->codec_header) {
    g_free (encoder->codec_header);
    encoder->codec_header = NULL;
    encoder->codec_header_size = 0;
    encoder->codec_header_sent = FALSE;
  }
}

static gboolean
gst_amlvenc_set_profile_and_level (GstAmlVEnc * encoder, GstCaps * caps)
{
  GstStructure *s;
  const gchar *profile;
  GstCaps *allowed_caps;
  GstStructure *s2;
  const gchar *allowed_profile;

  /* Constrained baseline is a strict subset of baseline. If downstream
   * wanted baseline and we produced constrained baseline, we can just
   * set the profile to baseline in the caps to make negotiation happy.
   * Same goes for baseline as subset of main profile and main as a subset
   * of high profile.
   */
  s = gst_caps_get_structure (caps, 0);
  profile = gst_structure_get_string (s, "profile");

  allowed_caps = gst_pad_get_allowed_caps (GST_VIDEO_ENCODER_SRC_PAD (encoder));

  if (allowed_caps == NULL)
    goto no_peer;

  if (!gst_caps_can_intersect (allowed_caps, caps)) {
    allowed_caps = gst_caps_make_writable (allowed_caps);
    allowed_caps = gst_caps_truncate (allowed_caps);
    s2 = gst_caps_get_structure (allowed_caps, 0);
    gst_structure_fixate_field_string (s2, "profile", profile);
    allowed_profile = gst_structure_get_string (s2, "profile");
    if (!g_strcmp0 (allowed_profile, "high")) {
      if (!g_strcmp0 (profile, "constrained-baseline")
          || !g_strcmp0 (profile, "baseline") || !g_strcmp0 (profile, "main")) {
        gst_structure_set (s, "profile", G_TYPE_STRING, "high", NULL);
        GST_INFO_OBJECT (encoder, "downstream requested high profile, but "
            "encoder will now output %s profile (which is a subset), due "
            "to how it's been configured", profile);
      }
    } else if (!g_strcmp0 (allowed_profile, "main")) {
      if (!g_strcmp0 (profile, "constrained-baseline")
          || !g_strcmp0 (profile, "baseline")) {
        gst_structure_set (s, "profile", G_TYPE_STRING, "main", NULL);
        GST_INFO_OBJECT (encoder, "downstream requested main profile, but "
            "encoder will now output %s profile (which is a subset), due "
            "to how it's been configured", profile);
      }
    } else if (!g_strcmp0 (allowed_profile, "baseline")) {
      if (!g_strcmp0 (profile, "constrained-baseline"))
        gst_structure_set (s, "profile", G_TYPE_STRING, "baseline", NULL);
    }
  }
  gst_caps_unref (allowed_caps);

no_peer:

  return TRUE;
}

/* gst_amlvenc_set_src_caps
 * Returns: TRUE on success.
 */
static gboolean
gst_amlvenc_set_src_caps (GstAmlVEnc * encoder, GstCaps * caps)
{
  GstCaps *outcaps;
  GstStructure *structure;
  GstVideoCodecState *state;
  GstTagList *tags;
  const gchar* mime_str = "video/x-h264";

  if (encoder->codec.id == CODEC_ID_H265) {
    mime_str = "video/x-h265";
  }
  outcaps = gst_caps_new_empty_simple (mime_str);
  structure = gst_caps_get_structure (outcaps, 0);

  gst_structure_set (structure, "stream-format", G_TYPE_STRING, "byte-stream",
      NULL);
  gst_structure_set (structure, "alignment", G_TYPE_STRING, "au", NULL);

  if (!gst_amlvenc_set_profile_and_level (encoder, outcaps)) {
    gst_caps_unref (outcaps);
    return FALSE;
  }

  state = gst_video_encoder_set_output_state (GST_VIDEO_ENCODER (encoder),
      outcaps, encoder->input_state);
  GST_DEBUG_OBJECT (encoder, "output caps: %" GST_PTR_FORMAT, state->caps);

  gst_video_codec_state_unref (state);

  tags = gst_tag_list_new_empty ();
  gst_tag_list_add (tags, GST_TAG_MERGE_REPLACE, GST_TAG_ENCODER, "v",
      GST_TAG_MAXIMUM_BITRATE, encoder->bitrate * 1000,
      GST_TAG_NOMINAL_BITRATE, encoder->bitrate * 1000, NULL);
  gst_video_encoder_merge_tags (GST_VIDEO_ENCODER (encoder), tags,
      GST_TAG_MERGE_REPLACE);
  gst_tag_list_unref (tags);

  return TRUE;
}

/* Wave521 multienc reports the original display-order frame index via
 * input_frame_num while emitting access units in decode order for B-frame GOPs.
 * Use matched input-frame PTS as display PTS, but generate monotonic DTS from
 * output order with a fixed initial lead so DTS stays <= PTS. */
static GstClockTime
gst_amlvenc_calculate_dts (GstAmlVEnc * encoder, GstVideoCodecFrame * frame, 
                           GstClockTime pts, gint frame_num)
{
  GstClockTime dts;

  if (!encoder->bframe_enabled || encoder->frame_duration == GST_CLOCK_TIME_NONE)
    return pts;

  switch (encoder->gop_pattern) {
    case 1: /* IBBBP */
    case 2: /* IBPBP */
    case 3: /* IBBB */
      break;
    default:
      return pts;
  }

  if (encoder->bframe_base_pts == GST_CLOCK_TIME_NONE)
    encoder->bframe_base_pts = pts;

  dts = encoder->bframe_base_pts + ((gint64) encoder->output_counter * encoder->frame_duration);

  GST_LOG_OBJECT (encoder,
      "Calculated B-frame DTS for display frame %d output_idx=%d pattern=%d: PTS=%" GST_TIME_FORMAT " DTS=%" GST_TIME_FORMAT,
      frame_num, encoder->output_counter, encoder->gop_pattern,
      GST_TIME_ARGS (pts), GST_TIME_ARGS (dts));

  return dts;
}

static GstClockTime
gst_amlvenc_adjust_bframe_pts (GstAmlVEnc *encoder, GstClockTime pts, gint frame_num)
{
  gint lead_frames = 0;

  if (!encoder->bframe_enabled || encoder->frame_duration == GST_CLOCK_TIME_NONE)
    return pts;

  switch (encoder->gop_pattern) {
    case 1:
      lead_frames = 5;
      break;
    case 2:
      lead_frames = 3;
      break;
    case 3:
      lead_frames = 4;
      break;
    default:
      return pts;
  }

  if (encoder->bframe_base_pts == GST_CLOCK_TIME_NONE)
    encoder->bframe_base_pts = pts;

  return encoder->bframe_base_pts +
      ((gint64) (frame_num + lead_frames) * encoder->frame_duration);
}

static GstVideoCodecFrame *
gst_amlvenc_get_output_frame (GstAmlVEnc *encoder, GstVideoEncoder *video_enc,
    const encoding_metadata_t *meta)
{
  GstVideoCodecFrame *frame = NULL;

  if (meta && meta->input_frame_num >= 0) {
    frame = gst_video_encoder_get_frame (video_enc, (guint32) meta->input_frame_num);
    if (frame) {
      GST_LOG_OBJECT (encoder,
          "matched output to input frame_num=%d type=%d",
          meta->input_frame_num, meta->extra.frame_type);
      return frame;
    }

    GST_WARNING_OBJECT (encoder,
        "failed to match output frame_num=%d, falling back to oldest pending frame",
        meta->input_frame_num);
  }

  return gst_video_encoder_get_oldest_frame (video_enc);
}

static void
gst_amlvenc_set_latency (GstAmlVEnc * encoder)
{
  GstVideoInfo *info = &encoder->input_state->info;
  gint max_delayed_frames;
  GstClockTime latency;

  /* GOP patterns: 0=IPP, 1=IBBBP, 2=IBPBP, 3=IBBB, 4=ALL_I
   * Calculate max_delayed_frames based on GOP pattern for B-frame support */
  switch (encoder->gop_pattern) {
    case 1: // IBBBP
      max_delayed_frames = 4;  // 3 B-frames + 1 reorder delay
      break;
    case 2: // IBPBP
      max_delayed_frames = 2;
      break;
    case 3: // IBBB
      max_delayed_frames = 3;
      break;
    default: // IPP, ALL_I
      max_delayed_frames = 0;
      break;
  }

  if (info->fps_n) {
    latency = gst_util_uint64_scale_ceil (GST_SECOND * info->fps_d,
        max_delayed_frames, info->fps_n);
  } else {
    /* FIXME: Assume 25fps. This is better than reporting no latency at
     * all and then later failing in live pipelines
     */
    latency = gst_util_uint64_scale_ceil (GST_SECOND * 1,
        max_delayed_frames, 25);
  }

  GST_INFO_OBJECT (encoder,
      "Updating latency to %" GST_TIME_FORMAT " (%d frames) for GOP pattern %d",
      GST_TIME_ARGS (latency), max_delayed_frames, encoder->gop_pattern);

  gst_video_encoder_set_latency (GST_VIDEO_ENCODER (encoder), latency, latency);
}

static gboolean
gst_amlvenc_set_format (GstVideoEncoder * video_enc,
    GstVideoCodecState * state)
{
  GstAmlVEnc *encoder = GST_AMLVENC (video_enc);
  GstVideoInfo *info = &state->info;
  GstCaps *template_caps;
  GstCaps *allowed_caps = NULL;
  const gchar* allowed_mime_name = NULL;

  /* If the encoder is initialized, do not reinitialize it again if not
   * necessary */
  if (encoder->codec.handle) {
    GstVideoInfo *old = &encoder->input_state->info;

    if (info->finfo->format == old->finfo->format
        && info->width == old->width && info->height == old->height
        && info->fps_n == old->fps_n && info->fps_d == old->fps_d
        && info->par_n == old->par_n && info->par_d == old->par_d
        && gst_video_colorimetry_is_equal (&info->colorimetry, &old->colorimetry)) {
      gst_video_codec_state_unref (encoder->input_state);
      encoder->input_state = gst_video_codec_state_ref (state);
      return TRUE;
    }
  }

  if (encoder->input_state)
    gst_video_codec_state_unref (encoder->input_state);

  encoder->input_state = gst_video_codec_state_ref (state);

  template_caps = gst_static_pad_template_get_caps (&src_factory);
  allowed_caps = gst_pad_get_allowed_caps (GST_VIDEO_ENCODER_SRC_PAD (encoder));

  if (allowed_caps && allowed_caps != template_caps && encoder->codec.id == CODEC_ID_NONE) {
    GstStructure *s;

    if (gst_caps_is_empty (allowed_caps)) {
      gst_caps_unref (allowed_caps);
      gst_caps_unref (template_caps);
      return FALSE;
    }

    allowed_caps = gst_caps_make_writable (allowed_caps);
    allowed_caps = gst_caps_fixate (allowed_caps);
    s = gst_caps_get_structure (allowed_caps, 0);
    allowed_mime_name = gst_structure_get_name (s);

    if (!g_strcmp0 (allowed_mime_name, "video/x-h265"))
    {
      encoder->codec.id = CODEC_ID_H265;
    } else {
      encoder->codec.id = CODEC_ID_H264;
    }

    gst_caps_unref (allowed_caps);
  }

  gst_caps_unref (template_caps);

  if (encoder->lossless_enable && encoder->codec.id == CODEC_ID_H264) {
    GST_ERROR_OBJECT (encoder, "lossless-enable is HEVC-only feature, not supported for H.264");
    return FALSE;
  }

  // init roi buffer info
  if (encoder->codec.id == CODEC_ID_H265) {
    encoder->roi.block_size = 32;
  }
  encoder->roi.buffer_info.width =
      (info->width + encoder->roi.block_size - 1) / encoder->roi.block_size;
  encoder->roi.buffer_info.height =
      (info->height + encoder->roi.block_size - 1) / encoder->roi.block_size;
  GST_DEBUG("info->width:%d, info->height:%d, roi_buffer_w:%d, roi_buffer_h:%d",
      info->width, info->height, encoder->roi.buffer_info.width, encoder->roi.buffer_info.height);
  if (encoder->roi.buffer_info.data == NULL) {
    encoder->roi.buffer_info.data =
        g_new(guchar, encoder->roi.buffer_info.width * encoder->roi.buffer_info.height );
    memset(encoder->roi.buffer_info.data,
           PROP_ROI_QUALITY_DEFAULT,
           encoder->roi.buffer_info.width * encoder->roi.buffer_info.height);
  }

  if (!gst_amlvenc_init_encoder (encoder))
    return FALSE;

  if (!gst_amlvenc_set_src_caps (encoder, state->caps)) {
    gst_amlvenc_close_encoder (encoder);
    return FALSE;
  }

  gst_amlvenc_set_latency (encoder);

  return TRUE;
}

static GstFlowReturn
gst_amlvenc_finish (GstVideoEncoder * video_enc)
{
  GstAmlVEnc *encoder = GST_AMLVENC (video_enc);

  GST_WARNING_OBJECT (encoder, "finish: EOS received, flushing encoder (stopping=%d)",
      encoder->stopping);

  /* Guard: if the encoder handle is already gone, nothing to flush */
  if (G_UNLIKELY (encoder->codec.handle == 0)) {
    GST_WARNING_OBJECT (encoder, "finish: codec handle is NULL, skipping flush");
    encoder->v10conv.pipeline_primed = FALSE;
    return GST_FLOW_OK;
  }

  /* Vulkan pipeline: encode the last buffered frame */
  if (encoder->v10conv.enabled && encoder->v10conv_backend == 0 &&
      encoder->v10conv.pipeline_primed) {

    GstVideoInfo *info = &encoder->input_state->info;

    /* The last frame's GPU result is in buf[1 - write_idx]
     * (write_idx was already flipped after the last GPU wait,
     * so the completed result is in the OTHER buffer).
     * Actually: write_idx points to where the NEXT GPU submit would go.
     * The last completed GPU result is in buf[1 - write_idx]. */
    gint last_buf = 1 - encoder->v10conv.write_idx;
    int last_fd = encoder->v10conv.output_buf[last_buf].fd;

    GST_WARNING_OBJECT(encoder, "Vulkan pipeline flush: encoding last buffered frame from buf[%d] fd=%d",
        last_buf, last_fd);

    vl_buffer_info_t inbuf_info;
    vl_buffer_info_t retbuf_info;
    memset(&inbuf_info, 0, sizeof(vl_buffer_info_t));
    inbuf_info.buf_type = DMA_TYPE;
    inbuf_info.buf_fmt = img_format_convert(GST_VIDEO_FORMAT_P010_10LE);
    inbuf_info.buf_stride = info->width * 2;
    inbuf_info.buf_info.dma_info.shared_fd[0] = last_fd;
    inbuf_info.buf_info.dma_info.shared_fd[1] = -1;
    inbuf_info.buf_info.dma_info.shared_fd[2] = -1;
    inbuf_info.buf_info.dma_info.num_planes = 1;

    g_mutex_lock (&encoder->encoder_lock);
    if (G_UNLIKELY (encoder->codec.handle == 0)) {
      g_mutex_unlock (&encoder->encoder_lock);
      GST_WARNING_OBJECT(encoder, "Vulkan pipeline flush: codec handle destroyed, skipping");
      encoder->v10conv.pipeline_primed = FALSE;
      return GST_FLOW_OK;
    }
    encoding_metadata_t meta =
        vl_multi_encoder_encode(encoder->codec.handle, FRAME_TYPE_AUTO,
                                encoder->codec.buf, &inbuf_info, &retbuf_info);
    g_mutex_unlock (&encoder->encoder_lock);

    if (meta.is_valid) {
        GstVideoCodecFrame *frame =
            gst_amlvenc_get_output_frame (encoder, video_enc, &meta);
      if (frame) {
        GstMapInfo map;
        frame->output_buffer = gst_video_encoder_allocate_output_buffer(
            video_enc, meta.encoded_data_length_in_bytes);
        gst_buffer_map(frame->output_buffer, &map, GST_MAP_WRITE);
        memcpy(map.data, encoder->codec.buf, meta.encoded_data_length_in_bytes);
        gst_buffer_unmap(frame->output_buffer, &map);

        if ((GST_CLOCK_TIME_NONE == GST_BUFFER_TIMESTAMP(frame->input_buffer))
            && info->fps_n && info->fps_d) {
          GST_BUFFER_TIMESTAMP(frame->input_buffer) = gst_util_uint64_scale(
              encoder->u4_first_pts_index++, GST_SECOND, info->fps_n / info->fps_d);
          GST_BUFFER_DURATION(frame->input_buffer) = gst_util_uint64_scale(
              1, GST_SECOND, info->fps_n / info->fps_d);
          frame->pts = GST_BUFFER_TIMESTAMP(frame->input_buffer);
        }
        frame->pts = gst_amlvenc_adjust_bframe_pts (encoder, frame->pts,
            meta.input_frame_num >= 0 ? meta.input_frame_num : 0);
        /* Use original input order for B-frame DTS handling */
        frame->dts = gst_amlvenc_calculate_dts (encoder, frame, frame->pts,
            meta.input_frame_num >= 0 ? meta.input_frame_num : encoder->dbg_frame_num);

        GST_WARNING_OBJECT(encoder, "Vulkan pipeline flush: finishing last frame pts=%" G_GINT64_FORMAT,
            (gint64)frame->pts);
        encoder->output_counter++;
        gst_video_encoder_finish_frame(video_enc, frame);
      }
    } else {
      GST_WARNING_OBJECT(encoder, "Vulkan pipeline flush: last frame encode returned invalid metadata");
    }

    encoder->v10conv.pipeline_primed = FALSE;
  }

  return GST_FLOW_OK;
}

static gboolean
gst_amlvenc_propose_allocation (GstVideoEncoder * encoder, GstQuery * query)
{
  GstAmlVEnc *self = GST_AMLVENC (encoder);

  if (!self->input_state)
    return FALSE;

  /*
   * Do not force a custom upstream pool/allocator from the encoder side.
   * Hardware decoder pipelines already negotiate their own DMABuf pool, and
   * pushing encoder-specific pool proposals upstream breaks hwdec->hwenc
   * allocation on T7. Keep negotiation minimal and only advertise video meta.
   */
  gst_query_add_allocation_meta (query, GST_VIDEO_META_API_TYPE, NULL);

  return GST_VIDEO_ENCODER_CLASS (parent_class)->propose_allocation (encoder,
      query);
}

/* chain function
 * this function does the actual processing
 */
static GstFlowReturn
gst_amlvenc_handle_frame (GstVideoEncoder * video_enc,
    GstVideoCodecFrame * frame)
{
  GstAmlVEnc *encoder = GST_AMLVENC (video_enc);
  GstFlowReturn ret;

  if (G_UNLIKELY (encoder->stopping)) {
    GST_WARNING_OBJECT (encoder, "handle_frame: stopping flag set, dropping frame");
    gst_video_codec_frame_unref (frame);
    return GST_FLOW_FLUSHING;
  }

  if (G_UNLIKELY (encoder->codec.handle == 0))
    goto not_inited;

  ret = gst_amlvenc_encode_frame (encoder, frame);

  /* input buffer is released later on */
  return ret;

/* ERRORS */
not_inited:
  {
    GST_WARNING_OBJECT (encoder, "Got buffer before set_caps was called");
    return GST_FLOW_NOT_NEGOTIATED;
  }
}

static GstFlowReturn
gst_amlvenc_encode_frame (GstAmlVEnc * encoder,
    GstVideoCodecFrame * frame)
{
  gint64 t_frame_start = g_get_monotonic_time();
  vl_frame_type_t frame_type = FRAME_TYPE_AUTO;
  GstVideoInfo *info = &encoder->input_state->info;
  GstMapInfo map;
  guint8 ui1_plane_num = 1;
  gint encode_data_len = -1;
  gint fd = -1;

  if (G_UNLIKELY (encoder->stopping)) {
    GST_WARNING_OBJECT (encoder, "encode_frame: stopping flag set, aborting");
    if (frame)
      gst_video_codec_frame_unref (frame);
    return GST_FLOW_FLUSHING;
  }

  if (G_UNLIKELY (encoder->codec.handle == 0)) {
    if (frame)
      gst_video_codec_frame_unref (frame);
    return GST_FLOW_NOT_NEGOTIATED;
  }

  if (frame) {
    if (GST_VIDEO_CODEC_FRAME_IS_FORCE_KEYFRAME (frame)) {
      GST_INFO_OBJECT (encoder, "Forcing key frame");
      frame_type = FRAME_TYPE_IDR;
    }
  }

  vl_buffer_info_t inbuf_info;
  vl_buffer_info_t retbuf_info;

  GstMemory *memory = gst_buffer_get_memory(frame->input_buffer, 0);
  guint n_mem = gst_buffer_n_memory (frame->input_buffer);
  gboolean is_dmabuf = gst_is_dmabuf_memory(memory);
  GstMapInfo minfo;
  GstVideoFormat submit_format = GST_VIDEO_INFO_FORMAT (info);
  encoder->fd[0] = -1;
  encoder->fd[1] = -1;
  encoder->fd[2] = -1;
  GST_DEBUG_OBJECT(encoder, "is_dmabuf[%d] width[%d] height[%d]",is_dmabuf,info->width,info->height);

  if (encoder->v10conv.enabled) {
      int in_fd = gst_dmabuf_memory_get_fd (memory);
      gst_memory_unref (memory);
      if (!is_dmabuf || in_fd < 0 || n_mem >= 2) {
        GST_ERROR_OBJECT (encoder, "internal v10 conversion requires single-plane dmabuf input");
        return GST_FLOW_ERROR;
      }
      if (!gst_amlvenc_prepare_v10conv_buffers (encoder, info)) {
        GST_ERROR_OBJECT (encoder, "failed to prepare internal v10 conversion buffers");
        return GST_FLOW_ERROR;
      }
      /* Use selected backend: 0=vulkan (default), 1=gles */
      gint64 t_conv_start = g_get_monotonic_time();
      
      if (encoder->v10conv_backend == 0) {
        /* Vulkan backend — double-buffered pipeline with 1-frame delay
         *
         * Frame 0 (priming): GPU convert to buf[0], return early (no encode).
         * Frame N (N>=1): GPU submit frame N to buf[N%2] (non-blocking),
         *   encode frame N-1 from buf[(N-1)%2] while GPU runs in parallel,
         *   wait for frame N's GPU.
         * EOS (finish): encode last buffered frame.
         *
         * Throughput: max(GPU, encoder) = ~13ms/frame instead of sum = ~21.5ms.
         */
        if (!encoder->v10conv.vulkan_ctx) {
          encoder->v10conv.vulkan_ctx = yuv422_vulkan_init(info->width, info->height);
          if (encoder->v10conv.vulkan_ctx) {
            GST_WARNING_OBJECT(encoder, "Vulkan converter initialized successfully");
          } else {
            GST_ERROR_OBJECT(encoder, "Vulkan init failed");
            return GST_FLOW_ERROR;
          }
        }

        gint cur = encoder->v10conv.write_idx;
        int out_fd = encoder->v10conv.output_buf[cur].fd;

        if (!encoder->v10conv.pipeline_primed) {
          /* === FRAME 0: Prime the pipeline === */
          /* Synchronous GPU convert, then return early without encoding.
           * The next handle_frame call will encode this result. */
          if (yuv422_vulkan_convert_dmabuf(encoder->v10conv.vulkan_ctx, in_fd, out_fd,
                  info->width, info->height) != 0) {
            GST_ERROR_OBJECT(encoder, "Vulkan conversion failed on first frame (%s)",
                yuv422_vulkan_last_error(encoder->v10conv.vulkan_ctx));
            return GST_FLOW_ERROR;
          }
          gint64 conv_us = g_get_monotonic_time() - t_conv_start;
          GST_WARNING_OBJECT(encoder, "[ENC-PROF] Vulkan conversion (priming, sync): %" G_GINT64_FORMAT " us (%.2f ms)",
              conv_us, conv_us / 1000.0);

          encoder->v10conv.pipeline_primed = TRUE;
          /* write_idx stays at cur — next frame will submit to buf[1-cur]
           * and encode from buf[cur] */
          encoder->v10conv.write_idx = 1 - cur;

          /* DON'T encode — drop this frame from GstVideoEncoder's pending
           * queue so its input buffer is released back to the upstream pool.
           * Previously we only called gst_video_codec_frame_unref() which
           * does NOT remove the frame from the pending list, causing the
           * input DMA-buf to stay pinned and eventually exhausting the
           * upstream buffer pool (pipeline stall after ~4 frames). */
          GST_WARNING_OBJECT(encoder, "Vulkan pipeline primed — dropping frame 0 (no encode)");
          frame->output_buffer = gst_buffer_new();
          GST_VIDEO_CODEC_FRAME_SET_DECODE_ONLY(frame);
          return gst_video_encoder_finish_frame(GST_VIDEO_ENCODER(encoder), frame);
        } else {
          /* === FRAME N (N>=1): Pipelined GPU + encode === */
          gint encode_buf = 1 - cur;  /* previous frame's GPU result */
          int encode_fd = encoder->v10conv.output_buf[encode_buf].fd;

          /* 1. Submit GPU for frame N (non-blocking) */
          if (yuv422_vulkan_convert_submit(encoder->v10conv.vulkan_ctx, in_fd, out_fd,
                  info->width, info->height) != 0) {
            GST_ERROR_OBJECT(encoder, "Vulkan submit failed (%s)",
                yuv422_vulkan_last_error(encoder->v10conv.vulkan_ctx));
            return GST_FLOW_ERROR;
          }

          gint64 conv_us = g_get_monotonic_time() - t_conv_start;
          GST_WARNING_OBJECT(encoder, "[ENC-PROF] Vulkan submit (pipelined): %" G_GINT64_FORMAT " us (%.2f ms)",
              conv_us, conv_us / 1000.0);

          /* 2. Set encoder to use PREVIOUS frame's GPU result.
           * The encoder will run for ~13ms, during which frame N's GPU
           * conversion runs in parallel on the Mali GPU. */
          encoder->fd[0] = encode_fd;
          encoder->fd[1] = -1;
          ui1_plane_num = 1;
          submit_format = GST_VIDEO_FORMAT_P010_10LE;

          /* Skip P010 stats logging for pipelined frames — go to encoder */
          goto v10conv_pipeline_encode;
        }
      } else {
        /* GLES backend */
        if (encoder->v10conv.output_y.fd_dup >= 0) {
          close(encoder->v10conv.output_y.fd_dup);
        }
        encoder->v10conv.output_y.fd_dup = dup(encoder->v10conv.output_y.fd);
        if (encoder->v10conv.output_y.fd_dup < 0) {
          GST_ERROR_OBJECT(encoder, "failed to dup output fd: %s", strerror(errno));
          return GST_FLOW_ERROR;
        }
        if (yuv422_gpu_gles_convert_p010_dmabuf (encoder->v10conv.gles_ctx, in_fd,
                encoder->v10conv.output_y.fd_dup,
                info->width, info->height) != 0) {
          GST_ERROR_OBJECT (encoder, "GLES conversion failed (%s)",
              yuv422_gpu_gles_last_error (encoder->v10conv.gles_ctx));
          close(encoder->v10conv.output_y.fd_dup);
          encoder->v10conv.output_y.fd_dup = -1;
          return GST_FLOW_ERROR;
        }
      }
      
      /* GLES profiling */
      {
        gint64 conv_us_gles = g_get_monotonic_time() - t_conv_start;
        GST_WARNING_OBJECT(encoder, "[ENC-PROF] GLES conversion: %" G_GINT64_FORMAT " us (%.2f ms)",
            conv_us_gles, conv_us_gles / 1000.0);
      }

      encoder->fd[0] = encoder->v10conv.output_y.fd;
      encoder->fd[1] = -1;
      ui1_plane_num = 1;
      submit_format = GST_VIDEO_FORMAT_P010_10LE;

      /* GPU shader outputs Wave521 MSB format directly - no CPU repack needed */
      GST_WARNING_OBJECT (encoder,
          "v10conv submit override: submit_format=%d (P010_10LE), width=%d, height=%d, stride=%d",
          submit_format, info->width, info->height, info->width * 2);

      {
        if (!encoder->logged_p010_stats) {
          /* Sync dmabuf for CPU read — invalidate cache to see GPU writes */
          struct dma_buf_sync sync_start = { .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ };
          struct dma_buf_sync sync_end = { .flags = DMA_BUF_SYNC_END | DMA_BUF_SYNC_READ };
          int sync_fd = encoder->fd[0];
          if (sync_fd >= 0) {
            ioctl(sync_fd, DMA_BUF_IOCTL_SYNC, &sync_start);
          }

          GstMapInfo p010_map;
          GstMemory *stats_mem = encoder->v10conv.output_buf[0].memory;
          if (stats_mem && gst_memory_map (stats_mem, &p010_map, GST_MAP_READ)) {
            guint y_samples = MIN ((guint)(info->width * info->height), 4096u);
            guint uv_samples = MIN ((guint)(info->width * info->height / 2), 4096u);
            guint y_min = 0xffff, y_max = 0;
            guint uv_min = 0xffff, uv_max = 0;
            guint i;
            const guint16 *p = (const guint16 *) p010_map.data;
            const guint16 *y = p;
            const guint16 *uv = p + (info->width * info->height);

            for (i = 0; i < y_samples; i++) {
              guint v = y[i];
              if (v < y_min) y_min = v;
              if (v > y_max) y_max = v;
            }
            for (i = 0; i < uv_samples; i++) {
              guint v = uv[i];
              if (v < uv_min) uv_min = v;
              if (v > uv_max) uv_max = v;
            }

            GST_WARNING_OBJECT (encoder,
                "internal P010 sample stats: Y[min=%u max=%u] UV[min=%u max=%u]",
                y_min, y_max, uv_min, uv_max);
            gst_memory_unmap (stats_mem, &p010_map);
            encoder->logged_p010_stats = TRUE;
          }

          if (sync_fd >= 0) {
            ioctl(sync_fd, DMA_BUF_IOCTL_SYNC, &sync_end);
          }
        }
      }

v10conv_pipeline_encode:
      /* Reached by: all v10conv paths. encoder->fd[0] is set to the correct output buffer. */
      ;
  } else if (is_dmabuf) {
      switch (GST_VIDEO_INFO_FORMAT(info)) {
      case GST_VIDEO_FORMAT_NV12:
      case GST_VIDEO_FORMAT_NV21:
      case GST_VIDEO_FORMAT_P010_10LE:
          {
          /* handle dma case scenario media convet encoder/hdmi rx encoder scenario*/
          encoder->fd[0] = gst_dmabuf_memory_get_fd(memory);
          gst_memory_unref(memory);
          if (n_mem >= 2) {
            GstMemory *memory_uv = gst_buffer_get_memory(frame->input_buffer, 1);
            encoder->fd[1] = gst_dmabuf_memory_get_fd(memory_uv);
            gst_memory_unref(memory_uv);
            ui1_plane_num = 2;
          } else {
            ui1_plane_num = 1;
          }
          break;
          }
      default: //hanle I420/YV12/RGB
        {
          /*
              Currently,For 420sp and RGB case,use one plane.
              420sp for usb camera case,usb camera y/u/v address is continious and no alignment requiremnet.
              Therefore,use one plane.
          */
          encoder->fd[0] = gst_dmabuf_memory_get_fd(memory);
          gst_memory_unref(memory);
          ui1_plane_num = 1;
          break;
        }
      }
  } else {
    gst_memory_unref(memory);
    /*
      non dmabuf case,due to encoder driver only handle dmabuf,so need convert to dma buffer case below.
     */ 
    if (encoder->imgproc.input.memory == NULL) {
      encoder->imgproc.input.memory =
        gst_allocator_alloc(encoder->dmabuf_alloc, info->size, NULL);
      if (encoder->imgproc.input.memory == NULL) {
        GST_DEBUG_OBJECT(encoder, "failed to allocate new dma buffer");
        return GST_FLOW_ERROR;
      }
      encoder->imgproc.input.fd = gst_dmabuf_memory_get_fd(encoder->imgproc.input.memory);
    }

    memory = encoder->imgproc.input.memory;
    fd = encoder->imgproc.input.fd;
    if (gst_memory_map(memory, &minfo, GST_MAP_WRITE)) {
      GstVideoFrame video_frame;
      gst_video_frame_map(&video_frame, info, frame->input_buffer, GST_MAP_READ);

      guint8 *pixel = GST_VIDEO_FRAME_PLANE_DATA (&video_frame, 0);
      memcpy(minfo.data, pixel, info->size);

      gst_video_frame_unmap (&video_frame);
      gst_memory_unmap(memory, &minfo);
    }
    encoder->fd[0] = fd;
  }
  /*
     For the rgb format,need convert to NV12 via ge2d.
     new imageproc handle when RGB case.
  */
  if (encoder->imgproc.handle) {
    if (encoder->dmabuf_alloc == NULL) {
      encoder->dmabuf_alloc = gst_amlion_allocator_obtain();
    }

    struct imgproc_buf inbuf, outbuf;
    inbuf.fd = encoder->fd[0];
    inbuf.is_ionbuf = gst_is_amlionbuf_memory(memory);

    if (encoder->imgproc.output.memory == NULL) {
      encoder->imgproc.output.memory = gst_allocator_alloc(
          encoder->dmabuf_alloc, encoder->imgproc.outbuf_size, NULL);
      if (encoder->imgproc.output.memory == NULL) {
        GST_ERROR_OBJECT(encoder, "failed to allocate new dma buffer");
        return GST_FLOW_ERROR;
      }
      encoder->imgproc.output.fd = gst_dmabuf_memory_get_fd(encoder->imgproc.output.memory);
    }

    fd = encoder->imgproc.output.fd;

    outbuf.fd = fd;
    outbuf.is_ionbuf = TRUE;

    struct imgproc_pos inpos = {
        0, 0, info->width, info->height, info->width, info->height};
    struct imgproc_pos outpos = {
        0, 0, info->width, info->height, info->width, info->height};

    imgproc_crop(encoder->imgproc.handle, inbuf, inpos,
                      GST_VIDEO_INFO_FORMAT(info), outbuf, outpos,
                      GST_VIDEO_FORMAT_NV12);

    encoder->fd[0] = fd;
  }

  memset(&inbuf_info, 0, sizeof(vl_buffer_info_t));
  inbuf_info.buf_type = DMA_TYPE;
  inbuf_info.buf_fmt = img_format_convert (submit_format);
  inbuf_info.buf_stride = (submit_format == GST_VIDEO_FORMAT_P010_10LE) ? info->width * 2 : GST_VIDEO_INFO_PLANE_STRIDE (info, 0);
  if (inbuf_info.buf_stride <= 0)
    inbuf_info.buf_stride = info->width;
  inbuf_info.buf_info.dma_info.shared_fd[0] = encoder->fd[0];
  inbuf_info.buf_info.dma_info.shared_fd[1] = encoder->fd[1];
  inbuf_info.buf_info.dma_info.shared_fd[2] = encoder->fd[2];
  inbuf_info.buf_info.dma_info.num_planes = ui1_plane_num;

  GST_WARNING_OBJECT (encoder,
      "ENCODER SUBMIT: submit_format=%s -> buf_fmt=%d stride=%d planes=%d fd0=%d fd1=%d width=%d height=%d",
      gst_video_format_to_string (submit_format),
      inbuf_info.buf_fmt, inbuf_info.buf_stride,
      inbuf_info.buf_info.dma_info.num_planes,
      inbuf_info.buf_info.dma_info.shared_fd[0],
      inbuf_info.buf_info.dma_info.shared_fd[1],
      info->width, info->height);

  /* Last chance to bail before entering the blocking encoder call */
  if (G_UNLIKELY (encoder->stopping)) {
    GST_WARNING_OBJECT (encoder, "encode_frame: stopping flag set before encoder call, aborting");
    if (frame)
      gst_video_codec_frame_unref (frame);
    return GST_FLOW_FLUSHING;
  }

  /* DMA_BUF_IOCTL_SYNC — bracket the VPU encode with read-access fences.
   * Without this, the vfm_cap exporter can recycle the underlying CMA pages
   * (via vf_put → VIDIOC_QBUF) while the Wave521 VPU is still reading,
   * causing a kernel crash.  The SYNC_START tells the exporter we are
   * actively reading; SYNC_END (after encode) releases that claim. */
  int sync_input_fd = encoder->fd[0];
  struct dma_buf_sync dbs_start = { .flags = DMA_BUF_SYNC_START | DMA_BUF_SYNC_READ };
  struct dma_buf_sync dbs_end   = { .flags = DMA_BUF_SYNC_END   | DMA_BUF_SYNC_READ };
  if (sync_input_fd >= 0) {
    int sr = ioctl(sync_input_fd, DMA_BUF_IOCTL_SYNC, &dbs_start);
    if (sr < 0) {
      GST_WARNING_OBJECT (encoder, "DMA_BUF_SYNC_START failed on fd %d: %s",
          sync_input_fd, g_strerror (errno));
    }
  }

  gint64 t_enc_start = g_get_monotonic_time();
  g_mutex_lock (&encoder->encoder_lock);
  if (G_UNLIKELY (encoder->codec.handle == 0)) {
    g_mutex_unlock (&encoder->encoder_lock);
    if (sync_input_fd >= 0)
      ioctl(sync_input_fd, DMA_BUF_IOCTL_SYNC, &dbs_end);
    GST_WARNING_OBJECT (encoder, "encode_frame: codec handle destroyed while waiting for lock");
    if (frame)
      gst_video_codec_frame_unref (frame);
    return GST_FLOW_FLUSHING;
  }
  encoding_metadata_t meta =
      vl_multi_encoder_encode(encoder->codec.handle, frame_type,
                              encoder->codec.buf, &inbuf_info, &retbuf_info);
  g_mutex_unlock (&encoder->encoder_lock);

  /* Release DMA-buf read access — exporter may now recycle the buffer */
  if (sync_input_fd >= 0) {
    int er = ioctl(sync_input_fd, DMA_BUF_IOCTL_SYNC, &dbs_end);
    if (er < 0) {
      GST_WARNING_OBJECT (encoder, "DMA_BUF_SYNC_END failed on fd %d: %s",
          sync_input_fd, g_strerror (errno));
    }
  }

  gint64 enc_us = g_get_monotonic_time() - t_enc_start;
  GST_WARNING_OBJECT(encoder, "[ENC-PROF] encoder: %" G_GINT64_FORMAT " us (%.2f ms)",
      enc_us, enc_us / 1000.0);

  /* Close GLES dup'd output fd after encoding completes */
  if (encoder->v10conv.enabled && encoder->v10conv_backend == 1 &&
      encoder->v10conv.output_y.fd_dup >= 0) {
    close(encoder->v10conv.output_y.fd_dup);
    encoder->v10conv.output_y.fd_dup = -1;
  }

  /* Vulkan pipeline: wait for current frame's GPU work (submitted earlier).
   * During the encoder call above (~13ms), the GPU was converting the NEXT
   * frame in parallel. By now (~13ms later) it should be done (only ~8.5ms). */
  if (encoder->v10conv.enabled && encoder->v10conv_backend == 0 &&
      encoder->v10conv.pipeline_primed) {
    gint64 t_gpu_wait = g_get_monotonic_time();
    if (yuv422_vulkan_convert_wait(encoder->v10conv.vulkan_ctx) != 0) {
      GST_ERROR_OBJECT(encoder, "Vulkan pipeline wait failed (%s)",
          yuv422_vulkan_last_error(encoder->v10conv.vulkan_ctx));
      /* Non-fatal — GPU result for NEXT frame is bad, but current encode succeeded */
    }
    gint64 gpu_wait_us = g_get_monotonic_time() - t_gpu_wait;
    GST_WARNING_OBJECT(encoder, "[ENC-PROF] Vulkan pipeline wait after encoder: %" G_GINT64_FORMAT " us (%.2f ms)",
        gpu_wait_us, gpu_wait_us / 1000.0);

    /* Flip double-buffer index — frame N's GPU result is now in buf[write_idx],
     * ready to be encoded next time. Next frame will write into buf[1-write_idx]. */
    encoder->v10conv.write_idx = 1 - encoder->v10conv.write_idx;
  }

  if (!meta.is_valid) {
    if (frame) {
      GST_ELEMENT_ERROR (encoder, STREAM, ENCODE, ("Encode v frame failed."),
          ("gst_amlvencoder_encode return code=%d", encode_data_len));
      gst_video_codec_frame_unref (frame);
      return GST_FLOW_ERROR;
    } else {
      return GST_FLOW_EOS;
    }
  }

  if (meta.encoded_data_length_in_bytes == 0) {
    GST_LOG_OBJECT (encoder,
        "encoder produced no output yet (reorder delay or delayed picture)");
    if (frame) {
      gst_video_codec_frame_unref (frame);
    }
    return GST_FLOW_OK;
  }

  if (frame) {
    gst_video_codec_frame_unref (frame);
  }

  frame = gst_amlvenc_get_output_frame (encoder, GST_VIDEO_ENCODER (encoder), &meta);
  if (!frame) {
    GST_ERROR_OBJECT (encoder, "No pending frame available after encoding");
    return GST_FLOW_ERROR;
  }

  guint out_size = meta.encoded_data_length_in_bytes;
  gboolean prepend_header = (encoder->bframe_enabled && !encoder->codec_header_sent &&
      encoder->codec_header && encoder->codec_header_size > 0);
  if (prepend_header) {
    out_size += encoder->codec_header_size;
  }

  frame->output_buffer = gst_video_encoder_allocate_output_buffer(
      GST_VIDEO_ENCODER(encoder), out_size);
  gst_buffer_map(frame->output_buffer, &map, GST_MAP_WRITE);
  if (prepend_header) {
    memcpy (map.data, encoder->codec_header, encoder->codec_header_size);
    memcpy (map.data + encoder->codec_header_size, encoder->codec.buf,
        meta.encoded_data_length_in_bytes);
    encoder->codec_header_sent = TRUE;
  } else {
    memcpy (map.data, encoder->codec.buf, meta.encoded_data_length_in_bytes);
  }
  gst_buffer_unmap (frame->output_buffer, &map);

  /*
  During encoder raw yuv file,and frame have no pts.
  so need fill it in order to avoid mux plugin fail.
  */
  if ((GST_CLOCK_TIME_NONE == GST_BUFFER_TIMESTAMP (frame->input_buffer))
      && info->fps_n && info->fps_d) {
      GST_LOG_OBJECT (encoder, "add for add pts end[%d] [%d]",info->fps_n,info->fps_d);
      GST_BUFFER_TIMESTAMP (frame->input_buffer) = gst_util_uint64_scale (encoder->u4_first_pts_index++, GST_SECOND, info->fps_n/info->fps_d);
      GST_BUFFER_DURATION (frame->input_buffer) = gst_util_uint64_scale (1, GST_SECOND, info->fps_n/info->fps_d);
      frame->pts = GST_BUFFER_TIMESTAMP (frame->input_buffer);

      //FIXME later for first_pts_index
      if (encoder->u4_first_pts_index == PTS_UINT_4_RESET) {
          GST_DEBUG_OBJECT (encoder, "PTS rollback");
          encoder->u4_first_pts_index = 0;
      }
  }

  frame->pts = gst_amlvenc_adjust_bframe_pts (encoder, frame->pts,
      meta.input_frame_num >= 0 ? meta.input_frame_num : 0);

  /* Use the new DTS calculation function for B-frame support
   * We use the dbg_frame_num counter as the frame number for timestamp calculation
   * This will correctly calculate DTS based on GOP pattern
   */
  frame->dts = gst_amlvenc_calculate_dts (encoder, frame, frame->pts,
      meta.input_frame_num >= 0 ? meta.input_frame_num : encoder->dbg_frame_num);

  GST_LOG_OBJECT (encoder,
      "output: dts %" G_GINT64_FORMAT " pts %" G_GINT64_FORMAT " (B-frame enabled: %d, GOP pattern: %d)",
      (gint64) frame->dts, (gint64) frame->pts, encoder->bframe_enabled, encoder->gop_pattern);

  encoder->output_counter++;

  if (meta.extra.frame_type == FRAME_TYPE_IDR || meta.extra.frame_type == FRAME_TYPE_I) {
    GST_DEBUG_OBJECT (encoder, "Output keyframe");
    GST_VIDEO_CODEC_FRAME_SET_SYNC_POINT (frame);
  } else {
    GST_VIDEO_CODEC_FRAME_UNSET_SYNC_POINT (frame);
  }

  {
    gint64 frame_us = g_get_monotonic_time() - t_frame_start;
    GST_WARNING_OBJECT(encoder, "[ENC-PROF] total frame: %" G_GINT64_FORMAT " us (%.2f ms)",
        frame_us, frame_us / 1000.0);
  }

  encoder->dbg_frame_num++;
  return gst_video_encoder_finish_frame ( GST_VIDEO_ENCODER(encoder), frame);

}

static void
gst_amlvenc_get_property (GObject * object, guint prop_id,
    GValue * value, GParamSpec * pspec)
{
  GstAmlVEnc *encoder = GST_AMLVENC (object);

  GST_OBJECT_LOCK (encoder);
  switch (prop_id) {
    case PROP_GOP:
      g_value_set_int (value, encoder->gop);
      break;
    case PROP_FRAMERATE:
      g_value_set_int (value, encoder->framerate);
      break;
    case PROP_BITRATE:
      g_value_set_int (value, encoder->bitrate);
      break;
    case PROP_MIN_BUFFERS:
      g_value_set_int (value, encoder->min_buffers);
      break;
    case PROP_MAX_BUFFERS:
      g_value_set_int (value, encoder->max_buffers);
      break;
    case PROP_ENCODER_BUFSIZE:
      g_value_set_int (value, encoder->encoder_bufsize / 1024);
      break;
    case PROP_ROI_ENABLED:
      g_value_set_boolean (value, encoder->roi.enabled);
      break;
    case PROP_ROI_ID:
      g_value_set_int (value, encoder->roi.id);
      break;
    case PROP_ROI_WIDTH: {
      struct RoiParamInfo *param_info = retrieve_roi_param_info(encoder, encoder->roi.id);
      g_value_set_float (value, param_info->location.width);
    } break;
    case PROP_ROI_HEIGHT: {
      struct RoiParamInfo *param_info = retrieve_roi_param_info(encoder, encoder->roi.id);
      g_value_set_float (value, param_info->location.height);
    } break;
    case PROP_ROI_X: {
      struct RoiParamInfo *param_info = retrieve_roi_param_info(encoder, encoder->roi.id);
      g_value_set_float (value, param_info->location.left);
    } break;
    case PROP_ROI_Y: {
      struct RoiParamInfo *param_info = retrieve_roi_param_info(encoder, encoder->roi.id);
      g_value_set_float (value, param_info->location.top);
    } break;
    case PROP_ROI_QUALITY: {
      struct RoiParamInfo *param_info = retrieve_roi_param_info(encoder, encoder->roi.id);
      g_value_set_int (value, param_info->quality);
    } break;
    case PROP_INTERNAL_BIT_DEPTH:
      g_value_set_int (value, encoder->internal_bit_depth);
      break;
    case PROP_GOP_PATTERN:
      g_value_set_int (value, encoder->gop_pattern);
      break;
    case PROP_RC_MODE:
      g_value_set_int (value, encoder->rc_mode);
      break;
    case PROP_LOSSLESS_ENABLE:
      g_value_set_boolean (value, encoder->lossless_enable);
      break;
    case PROP_QP_B:
      g_value_set_int (value, encoder->qp_b);
      break;
    case PROP_MIN_QP_B:
      g_value_set_int (value, encoder->min_qp_b);
      break;
    case PROP_MAX_QP_B:
      g_value_set_int (value, encoder->max_qp_b);
      break;
    case PROP_V10CONV_BACKEND:
      g_value_set_int (value, encoder->v10conv_backend);
      break;
    case PROD_ENABLE_DMALLOCATOR:
      g_value_set_boolean (value, encoder->b_enable_dmallocator);
      break;
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }
  GST_OBJECT_UNLOCK (encoder);
}

static void
gst_amlvenc_set_property (GObject * object, guint prop_id,
    const GValue * value, GParamSpec * pspec)
{
  GstAmlVEnc *encoder = GST_AMLVENC (object);
  gboolean roi_set_flag = false;

  GST_OBJECT_LOCK (encoder);

  switch (prop_id) {
    case PROP_GOP:
      encoder->gop = g_value_get_int (value);
      break;
    case PROP_FRAMERATE:
      encoder->framerate = g_value_get_int (value);
      break;
    case PROP_BITRATE:
      encoder->bitrate = g_value_get_int (value);
      break;
    case PROP_MIN_BUFFERS:
      encoder->min_buffers = g_value_get_int (value);
      break;
    case PROP_MAX_BUFFERS:
      encoder->max_buffers = g_value_get_int (value);
      break;
    case PROP_ENCODER_BUFSIZE:
      encoder->encoder_bufsize = g_value_get_int (value) * 1024;
      break;
    case PROP_ROI_ENABLED: {
      gboolean enabled = g_value_get_boolean (value);
      if (!enabled) {
        cleanup_roi_param_list(encoder);
        memset(encoder->roi.buffer_info.data,
               PROP_ROI_QUALITY_DEFAULT,
               encoder->roi.buffer_info.width * encoder->roi.buffer_info.height);
      }
      if (enabled != encoder->roi.enabled) {
        encoder->roi.enabled = enabled;
        roi_set_flag = true;
      }
    } break;
    case PROP_ROI_ID: {
      gint id = g_value_get_int (value);
      if (id != encoder->roi.id) {
        encoder->roi.id = id;
      }
    } break;
    case PROP_ROI_WIDTH: {
      struct RoiParamInfo *param_info = retrieve_roi_param_info(encoder, encoder->roi.id);
      param_info->location.width = g_value_get_float (value);
    } break;
    case PROP_ROI_HEIGHT: {
      struct RoiParamInfo *param_info = retrieve_roi_param_info(encoder, encoder->roi.id);
      param_info->location.height = g_value_get_float (value);
    } break;
    case PROP_ROI_X: {
      struct RoiParamInfo *param_info = retrieve_roi_param_info(encoder, encoder->roi.id);
      param_info->location.left = g_value_get_float (value);
    } break;
    case PROP_ROI_Y: {
      struct RoiParamInfo *param_info = retrieve_roi_param_info(encoder, encoder->roi.id);
      param_info->location.top = g_value_get_float (value);
    } break;
    case PROP_ROI_QUALITY: {
      gint quality = g_value_get_int (value);
      struct RoiParamInfo *param_info = retrieve_roi_param_info(encoder, encoder->roi.id);
      if (quality != param_info->quality) {
        param_info->quality = quality;
        roi_set_flag = true;
      }
    } break;
    case PROP_INTERNAL_BIT_DEPTH:
      encoder->internal_bit_depth = g_value_get_int (value);
      break;
    case PROP_GOP_PATTERN:
      encoder->gop_pattern = g_value_get_int (value);
      break;
    case PROP_RC_MODE:
      encoder->rc_mode = g_value_get_int (value);
      break;
    case PROP_LOSSLESS_ENABLE:
      encoder->lossless_enable = g_value_get_boolean (value);
      break;
    case PROP_QP_B:
      encoder->qp_b = g_value_get_int (value);
      GST_LOG_OBJECT (encoder, "qp-b set to %d", encoder->qp_b);
      break;
    case PROP_MIN_QP_B:
      encoder->min_qp_b = g_value_get_int (value);
      GST_LOG_OBJECT (encoder, "min-qp-b set to %d", encoder->min_qp_b);
      break;
    case PROP_MAX_QP_B:
      encoder->max_qp_b = g_value_get_int (value);
      GST_LOG_OBJECT (encoder, "max-qp-b set to %d", encoder->max_qp_b);
      break;
    case PROP_V10CONV_BACKEND:
      encoder->v10conv_backend = g_value_get_int (value);
      GST_WARNING_OBJECT (encoder, "v10conv-backend set to %d (0=vulkan, 1=gles)",
          encoder->v10conv_backend);
      break;
    case PROD_ENABLE_DMALLOCATOR: {
      gboolean enabled = g_value_get_boolean (value);
      encoder->b_enable_dmallocator = enabled;
      break;
    }
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID (object, prop_id, pspec);
      break;
  }

  if (roi_set_flag) {
    if (encoder->roi.srcid)
      g_source_remove (encoder->roi.srcid);
    encoder->roi.srcid = g_idle_add((GSourceFunc)idle_set_roi, encoder);
  }

  GST_OBJECT_UNLOCK (encoder);
  return;
}

static gboolean
amlvenc_init (GstPlugin * amlvenc)
{
  GST_DEBUG_CATEGORY_INIT (gst_amlvenc_debug, "amlvenc", 0,
      "amlogic h264/h265 encoding element");

  return gst_element_register (amlvenc, "amlvenc", GST_RANK_PRIMARY,
      GST_TYPE_AMLVENC);
}

#ifndef VERSION
#define VERSION "1.0.0"
#endif

#ifndef PACKAGE
#define PACKAGE "aml_package"
#endif
#ifndef PACKAGE_NAME
#define PACKAGE_NAME "aml_package"
#endif
#ifndef GST_PACKAGE_ORIGIN
#define GST_PACKAGE_ORIGIN "http://amlogic.com/"
#endif

GST_PLUGIN_DEFINE (
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    amlvenc,
    "Amlogic h264/h265 encoder plugins",
    amlvenc_init,
    VERSION,
    "LGPL",
    "amlogic h264/h265 ecoding",
    "http://openlinux.amlogic.com"
)
