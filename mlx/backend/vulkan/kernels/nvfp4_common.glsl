// Shared NVFP4 decode helpers for gather/dense matmul shaders.
#ifndef MLX_NVFP4_COMMON_GLSL
#define MLX_NVFP4_COMMON_GLSL

float nvfp4_fp8_e4m3_to_fp32(uint8_t x) {
  uint v = (uint(x) & 127u) << 7;
  float val = unpackHalf2x16(v).x * 256.0;
  return (x & uint8_t(128u)) != 0u ? -val : val;
}

float nvfp4_lut(uint q) {
  const float lut[16] = float[16](
      +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0, -0.0, -0.5, -1.0, -1.5,
      -2.0, -3.0, -4.0, -6.0);
  return lut[q];
}

float nvfp4_decode_nibble(uint packed, uint lane, float scale) {
  return nvfp4_lut((packed >> (lane * 4u)) & 0xFu) * scale;
}

#endif
