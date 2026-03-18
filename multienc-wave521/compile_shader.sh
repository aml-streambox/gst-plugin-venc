#!/bin/bash
# Compile GLSL compute shader to SPIR-V and generate a C header.
#
# This script produces yuv422_to_p010_spv.h which is #include'd by
# yuv422_converter_vulkan.c.  You MUST re-run this script after any
# change to yuv422_to_p010.comp, then rebuild the plugin:
#
#   bash compile_shader.sh
#   rm -f yuv422_converter_vulkan.o   # force recompile
#   bitbake gst-plugin-venc-multienc -f -c compile
#
# Requires glslangValidator (from KhronosGroup/glslang) or glslc
# (from the Vulkan SDK), plus xxd.

INPUT="yuv422_to_p010.comp"
OUTPUT="yuv422_to_p010.spv"

if [ ! -f "$INPUT" ]; then
    echo "Error: $INPUT not found"
    exit 1
fi

if command -v glslangValidator &> /dev/null; then
    glslangValidator -V "$INPUT" -o "$OUTPUT"
    echo "Compiled $INPUT -> $OUTPUT"
elif command -v glslc &> /dev/null; then
    glslc "$INPUT" -o "$OUTPUT"
    echo "Compiled $INPUT -> $OUTPUT"
else
    echo "Error: Neither glslangValidator nor glslc found"
    echo "Install Vulkan SDK to compile shaders"
    exit 1
fi

# Convert SPIR-V to C header for embedding
# Use underscore in filename to match what the C code includes
HEADER_OUTPUT="yuv422_to_p010_spv.h"
xxd -i "$OUTPUT" > "$HEADER_OUTPUT"
echo "Generated $HEADER_OUTPUT"
