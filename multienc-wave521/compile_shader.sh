#!/bin/bash
# Compile GLSL compute shader to SPIR-V
# Requires glslangValidator from KhronosGroup/glslang

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
