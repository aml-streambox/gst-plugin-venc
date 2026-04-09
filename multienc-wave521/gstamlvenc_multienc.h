/*
 * Copyright (C) 2014-2019 Amlogic, Inc. All rights reserved.
 *
 * All information contained herein is Amlogic confidential.
 *
 */

#ifndef __GST_AMLVENC_H__
#define __GST_AMLVENC_H__

#include <gst/gst.h>
#include <gst/gstallocator.h>
#include <gst/video/video.h>
#include <gst/video/gstvideoencoder.h>
//#include <list.h>

#include "list.h"
#include "vp_multi_codec_1_0.h"

/* Forward declarations for per-instance GPU converter contexts */
typedef struct VulkanCtx VulkanCtx;
typedef struct GpuCtx GpuCtx;

#ifdef HAVE_STDINT_H
#include <stdint.h>
#endif
//#include <vp_multi_codec_1_0.h>

#define PTS_UINT_4_RESET 4294967295

G_BEGIN_DECLS

#define GST_TYPE_AMLVENC \
  (gst_amlvenc_get_type())
#define GST_AMLVENC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_AMLVENC,GstAmlVEnc))
#define GST_AMLVENC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_AMLVENC,GstAmlVEncClass))
#define GST_IS_AMLVENC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_AMLVENC))
#define GST_IS_AMLVENC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_AMLVENC))

typedef struct _GstAmlVEnc      GstAmlVEnc;
typedef struct _GstAmlVEncClass GstAmlVEncClass;
typedef struct _GstAmlVEncVTable GstAmlVEncVTable;

struct _GstAmlVEnc
{
  GstVideoEncoder element;

  /*< private >*/
  struct codec_info {
    vl_codec_handle_t handle;
    vl_codec_id_t id;
    guchar *buf;
  } codec;

  struct imgproc_info {
    void *handle;
    gint outbuf_size;
    gint width;
    gint height;
    struct {
      GstMemory *memory;
      gint fd;
    } input, output;
  } imgproc;

  struct v10conv_info {
    gboolean enabled;
    GstAllocator *allocator;
    struct {
      GstMemory *memory;
      gint fd;
      gint fd_dup;  /* dup'd fd we own, valid during encoding */
    } output_y, output_uv;
    /* Double-buffered output for pipelined GPU+encoder */
    struct {
      GstMemory *memory;
      gint fd;
    } output_buf[2];
    gint write_idx;       /* next buffer for GPU to write into (0 or 1) */
    gboolean pipeline_primed; /* TRUE after first frame's GPU work submitted */
    VulkanCtx *vulkan_ctx;  /* per-instance Vulkan converter (NULL = not initialized) */
    GpuCtx *gles_ctx;      /* per-instance GLES converter (NULL = not initialized) */
  } v10conv;

  GstAllocator *dmabuf_alloc;

  /* properties */
  gint fd[3];
  gint gop;
  gint framerate;
  guint bitrate;
  guint min_buffers;
  guint max_buffers;
  guint encoder_bufsize;
  guint u4_first_pts_index;
  gboolean b_enable_dmallocator;

  /* advanced encoding properties */
  gint internal_bit_depth;
  gint gop_pattern;
  gint rc_mode;
  gboolean lossless_enable;
  
  /* B-frame QP properties */
  gint qp_i;
  gint qp_p;
  gint qp_b;
  gint min_qp_b;
  gint max_qp_b;
  
  /* v10 conversion backend: 0=vulkan (default), 1=gles */
  gint v10conv_backend;
  
  /* B-frame reordering state */
  gboolean bframe_enabled;
  gint gop_size;
  gint output_counter;
  GstClockTime frame_duration;
  GstClockTime bframe_base_pts;

  struct roi_info {
    guint srcid;
    gboolean enabled;
    gint id;
    gint block_size;
    struct listnode param_info;
    struct _buffer_info {
      gint width;
      gint height;
      guchar *data;
    } buffer_info;
  } roi;

  /* input description */
  GstVideoCodecState *input_state;

  /* abort/shutdown mechanism for clean pipeline teardown */
  volatile gboolean stopping;
  GMutex encoder_lock;  /* protects codec.handle access during shutdown */

  /* SIGINT-driven EOS: encoder self-injects EOS on Ctrl+C */
  guint sigint_source_id;

  /* per-instance debug/profiling state (was static local) */
  unsigned long dbg_frame_num;
  gboolean logged_p010_stats;

  /* CBR filler accounting */
  guint64 cbr_target_bytes_scaled;
  guint64 cbr_emitted_bytes;

};

struct _GstAmlVEncClass
{
  GstVideoEncoderClass parent_class;
};

GType gst_amlvenc_get_type (void);

G_END_DECLS

#endif /* __GST_AMLVENC_H__ */
