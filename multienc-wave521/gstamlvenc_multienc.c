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
#include <string.h>

#include "gstamlvenc_multienc.h"
#include "imgproc.h"

#include "gstamlionallocator.h"

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
#define DRMBP_EXTRA_BUF_SIZE_FOR_DISPLAY 1
#define DRMBP_LIMIT_MAX_BUFSIZE_TO_BUFSIZE 1

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
  PROD_ENABLE_DMALLOCATOR
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

  g_object_class_install_property (gobject_class, PROD_ENABLE_DMALLOCATOR,
      g_param_spec_boolean ("enable-dmallocator", "enable-dmallocator", "Enable/Disable dmallocator",
          FALSE,
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
}

static void
gst_amlvenc_finalize (GObject * object)
{
  G_OBJECT_CLASS (parent_class)->finalize (object);
}


static gboolean
gst_amlvenc_start (GstVideoEncoder * encoder)
{
  GstAmlVEnc *venc = GST_AMLVENC (encoder);

  venc->dmabuf_alloc = gst_amlion_allocator_obtain();

  if (venc->codec.buf == NULL) {
    venc->codec.buf = g_new (guchar, venc->encoder_bufsize);
  }
  venc->imgproc.input.memory = NULL;
  venc->imgproc.output.memory = NULL;
  /* make sure that we have enough time for first DTS,
     this is probably overkill for most streams */
  gst_video_encoder_set_min_pts (encoder, GST_MSECOND * 30);

  return TRUE;
}

static gboolean
gst_amlvenc_stop (GstVideoEncoder * encoder)
{
  GstAmlVEnc *venc = GST_AMLVENC (encoder);

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

  encoder->codec.handle = vl_multi_encoder_init(encoder->codec.id, encode_info, &qp_tbl);

  if (encoder->codec.handle == 0) {
    GST_ELEMENT_ERROR (encoder, STREAM, ENCODE,
        ("Can not initialize v encoder."), (NULL));
    return FALSE;
  }

  if (!gst_amlvenc_set_roi (encoder)) {
    return FALSE;
  }

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
  if (encoder->codec.handle != 0) {
    vl_multi_encoder_destroy(encoder->codec.handle);
    encoder->codec.handle = 0;
  }
  if (encoder->imgproc.handle) {
    imgproc_deinit(encoder->imgproc.handle);
    encoder->imgproc.handle = NULL;
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

static void
gst_amlvenc_set_latency (GstAmlVEnc * encoder)
{
  GstVideoInfo *info = &encoder->input_state->info;
  gint max_delayed_frames;
  GstClockTime latency;

  max_delayed_frames = 0;

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
      "Updating latency to %" GST_TIME_FORMAT " (%d frames)",
      GST_TIME_ARGS (latency), max_delayed_frames);

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
        && info->par_n == old->par_n && info->par_d == old->par_d) {
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
        g_new(guchar, encoder->roi.buffer_info.width * encoder->roi.buffer_info.width );
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
gst_amlvenc_finish (GstVideoEncoder * encoder)
{
  return GST_FLOW_OK;
}

static gboolean
gst_amlvenc_propose_allocation (GstVideoEncoder * encoder, GstQuery * query)
{
  GstAmlVEnc *self = GST_AMLVENC (encoder);
  GstVideoInfo *info;
  guint size, min = 0, max = 0;
  GstCaps *caps;
  GstBufferPool *pool = NULL;
  gboolean need_pool = FALSE;

  if (!self->input_state)
    return FALSE;

  info = &self->input_state->info;

  if (info) {
      switch (GST_VIDEO_INFO_FORMAT(info)) {
      case GST_VIDEO_FORMAT_NV12:
      case GST_VIDEO_FORMAT_NV21:
          {
            GST_DEBUG_OBJECT(encoder, "choose gst_drm_bufferpool");
            gst_query_parse_allocation(query, &caps, &need_pool);
            GST_DEBUG_OBJECT(encoder, "need_pool: %d", need_pool);

              if (need_pool) {
                      pool = gst_drm_bufferpool_new(FALSE, GST_DRM_BUFFERPOOL_TYPE_VIDEO_PLANE);
                      GST_DEBUG_OBJECT(encoder, "new gst_drm_bufferpool");
                  }

              gst_query_add_allocation_pool(query, pool, info->size, DRMBP_EXTRA_BUF_SIZE_FOR_DISPLAY, DRMBP_LIMIT_MAX_BUFSIZE_TO_BUFSIZE);
              GST_DEBUG_OBJECT(encoder, "info->size: %d", info->size);
                      if (pool)
                      g_object_unref(pool);

              gst_query_add_allocation_meta (query, GST_VIDEO_META_API_TYPE, NULL);

          break;
          }
      default: //hanle not NV12/NV21
          {
            GST_DEBUG_OBJECT(encoder, "choose fake bufferpool");
            if (gst_query_get_n_allocation_pools (query) > 0) {
              gst_query_parse_nth_allocation_pool (query, 0, NULL, &size, &min, &max);
              size = MAX (size, info->size);
              gst_query_set_nth_allocation_pool (query, 0, NULL, size, self->min_buffers, self->max_buffers);
            } else {
              gst_query_add_allocation_pool (query, NULL, info->size, self->min_buffers, self->max_buffers);
              GST_DEBUG_OBJECT(encoder, "info->size: %d", info->size);
            }

              gst_query_add_allocation_meta (query, GST_VIDEO_META_API_TYPE, NULL);
          break;
          }
      }
  } else {
        GST_DEBUG_OBJECT(encoder, "can not get videoinfo");
        return FALSE;
  }

 if (self->b_enable_dmallocator) {
    GstAllocator *allocator = NULL;
    GstAllocationParams params;
    /* we got configuration from our peer or the decide_allocation method,
     * parse them */
    if (gst_query_get_n_allocation_params (query) > 0) {
       gst_query_parse_nth_allocation_param (query, 0, &allocator, &params);
    } else {
      allocator = NULL;
      gst_allocation_params_init (&params);
    }

    /* For the camera + jpegdec + encoder case,jpegdec use videobuffer pool which is software allocate.
    caused cp memory to dmabuffer during encoder frame.
    currently,provide the dmaallocator to upstreamer element to avoid copy(optimize)
    Try to update allocator*/
    if (self->dmabuf_alloc)
      gst_query_add_allocation_param (query, self->dmabuf_alloc, &params);

    if (allocator)
      gst_object_unref (allocator);
  }

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
  vl_frame_type_t frame_type = FRAME_TYPE_AUTO;
  GstVideoInfo *info = &encoder->input_state->info;
  GstMapInfo map;
  guint8 ui1_plane_num = 1;
  gint encode_data_len = -1;
  gint fd = -1;

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
  gboolean is_dmabuf = gst_is_dmabuf_memory(memory);
  GstMapInfo minfo;
  GST_DEBUG_OBJECT(encoder, "is_dmabuf[%d] width[%d] height[%d]",is_dmabuf,info->width,info->height);

  if (is_dmabuf) {
      switch (GST_VIDEO_INFO_FORMAT(info)) {
      case GST_VIDEO_FORMAT_NV12:
      case GST_VIDEO_FORMAT_NV21:
          {
          /* handle dma case scenario media convet encoder/hdmi rx encoder scenario*/
          encoder->fd[0] = gst_dmabuf_memory_get_fd(memory);
          gst_memory_unref(memory);
          GstMemory *memory_uv = gst_buffer_get_memory(frame->input_buffer, 1);
          encoder->fd[1] = gst_dmabuf_memory_get_fd(memory_uv);
          gst_memory_unref(memory_uv);
          ui1_plane_num = 2;
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
  inbuf_info.buf_info.dma_info.shared_fd[0] = encoder->fd[0];
  inbuf_info.buf_info.dma_info.shared_fd[1] = encoder->fd[1];
  inbuf_info.buf_info.dma_info.shared_fd[2] = encoder->fd[2];
  inbuf_info.buf_info.dma_info.num_planes = ui1_plane_num;

  encoding_metadata_t meta =
      vl_multi_encoder_encode(encoder->codec.handle, frame_type,
                              encoder->codec.buf, &inbuf_info, &retbuf_info);

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

  if (frame) {
    gst_video_codec_frame_unref (frame);
  }

  //frame = gst_video_encoder_get_frame (GST_VIDEO_ENCODER (encoder), input_frame->system_frame_number);
  frame = gst_video_encoder_get_oldest_frame (GST_VIDEO_ENCODER (encoder));
  if (!frame) {
    gst_video_codec_frame_unref (frame);
    return GST_FLOW_ERROR;
  }

  frame->output_buffer = gst_video_encoder_allocate_output_buffer(
      GST_VIDEO_ENCODER(encoder), meta.encoded_data_length_in_bytes);
  gst_buffer_map(frame->output_buffer, &map, GST_MAP_WRITE);
  memcpy (map.data, encoder->codec.buf, meta.encoded_data_length_in_bytes);
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

  frame->dts = frame->pts;

  GST_LOG_OBJECT (encoder,
      "output: dts %" G_GINT64_FORMAT " pts %" G_GINT64_FORMAT,
      (gint64) frame->dts, (gint64) frame->pts);

  if (frame_type == FRAME_TYPE_IDR) {
    GST_DEBUG_OBJECT (encoder, "Output keyframe");
    GST_VIDEO_CODEC_FRAME_SET_SYNC_POINT (frame);
  } else {
    GST_VIDEO_CODEC_FRAME_UNSET_SYNC_POINT (frame);
  }

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
