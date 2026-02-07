/*
 * Copyright (C) 2014-2019 Amlogic, Inc. All rights reserved.
 *
 * All information contained herein is Amlogic confidential.
 *
 */

#ifndef __GST_AMLVENC_H__
#define __GST_AMLVENC_H__

#include <gst/gst.h>
#include <gst/video/video.h>
#include <gst/video/gstvideoencoder.h>
//#include <list.h>

#include "list.h"
#include "vp_multi_codec_1_0.h"

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

};

struct _GstAmlVEncClass
{
  GstVideoEncoderClass parent_class;
};

GType gst_amlvenc_get_type (void);

G_END_DECLS

#endif /* __GST_AMLVENC_H__ */
