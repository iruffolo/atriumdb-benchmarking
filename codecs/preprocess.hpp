#pragma once

// preprocess.hpp
// ==============
// Preprocessing transforms for time-series compression.
//
// Provides reversible transforms (DPCM, delta-of-delta, XOR, byte shuffle,
// raw) that convert int64_t sample sequences into byte buffers suitable for
// compression. Each transform defines an encode/decode pair and can be
// combined with any backend compressor (Zstd, Bzip2, LZMA, etc.).
//

#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

namespace preprocess {

// Enum to define what preprocess mode to use
enum class Mode {
  RAW,
  DPCM,
  DOD,
  XOR,
  BYTE_SHUFFLE,
};

// ═══════════════════════════════════════════════════════════════════════════
//  Raw encode
// ═══════════════════════════════════════════════════════════════════════════
//
std::vector<uint8_t> raw_encode(std::span<const int64_t> s) {
  std::vector<uint8_t> out(s.size() * sizeof(int64_t));
  std::memcpy(out.data(), s.data(), out.size());
  return out;
}

std::vector<int64_t> raw_decode(std::span<const uint8_t> data, size_t n) {
  std::vector<int64_t> out(n);
  std::memcpy(out.data(), data.data(), n * sizeof(int64_t));
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Byte shuffle — byte-lane transpose (HDF5 / Blosc style)
// ═══════════════════════════════════════════════════════════════════════════
//
//  Each int64 is 8 bytes.  A raw stream interleaves them:
//    [b0 b1 b2 b3 b4 b5 b6 b7 | b0 b1 b2 b3 b4 b5 b6 b7 | ...]
//                   value 0                  value 1
//
//  After shuffling all bytes at position k are grouped together:
//    [b0_0 b0_1 b0_2 ... | b1_0 b1_1 ... | ... | b7_0 b7_1 ...]
//      lane 0 (n bytes)     lane 1 (n bytes)      lane 7 (n bytes)
//
//  For physiological signals the high-order bytes (lanes 4–7 for little-
//  endian int64) change very slowly — they become highly repetitive and
//  compress very well.  The low-order bytes vary more but are also denser
//  in information, so the net effect is a better overall ratio.
//
//  The output is a flat byte buffer of length n * sizeof(int64_t).
//  ELEM_BYTES is a template parameter so the same code works for int32_t,
//  float, double etc. if needed in the future.
//
std::vector<uint8_t> shuffle_encode(std::span<const int64_t> s) {
  size_t n = s.size();
  size_t B = sizeof(int64_t);

  std::vector<uint8_t> out(n * B);
  auto *src = reinterpret_cast<const uint8_t *>(s.data());

  for (size_t b = 0; b < B; ++b)
    for (size_t i = 0; i < n; ++i)
      out[b * n + i] = src[i * B + b];

  return out;
}

std::vector<int64_t> shuffle_decode(std::span<const uint8_t> data, size_t n) {
  size_t B = sizeof(int64_t);
  std::vector<int64_t> out(n);

  auto *dst = reinterpret_cast<uint8_t *>(out.data());

  for (size_t b = 0; b < B; ++b)
    for (size_t i = 0; i < n; ++i)
      dst[i * B + b] = data[b * n + i];

  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Delta-of-delta — second-order differencing
// ═══════════════════════════════════════════════════════════════════════════
//
//  Forward:  d1[0] = samples[0],  d1[i] = samples[i] - samples[i-1]
//            d2[0] = d1[0],       d2[i] = d1[i]      - d1[i-1]
//
//  Inverse:  cumsum(d2) → d1,  cumsum(d1) → original
//
//  Best for signals with a slowly changing first derivative (e.g. smoothly
//  sweeping ECG baselines, linearly drifting sensors).  For step-change
//  signals it may be worse than plain DPCM.
std::vector<uint8_t> dod_encode(std::span<const int64_t> s) {
  size_t n = s.size();
  std::vector<int64_t> d1(n), d2(n);

  d1[0] = s[0];
  for (size_t i = 1; i < n; ++i)
    d1[i] = s[i] - s[i - 1];

  d2[0] = d1[0];
  for (size_t i = 1; i < n; ++i)
    d2[i] = d1[i] - d1[i - 1];

  std::vector<uint8_t> out(n * sizeof(int64_t));
  std::memcpy(out.data(), d2.data(), out.size());
  return out;
}

std::vector<int64_t> dod_decode(std::span<const uint8_t> data, size_t n) {
  auto *d2 = reinterpret_cast<const int64_t *>(data.data());

  std::vector<int64_t> tmp(d2, d2 + n);
  std::vector<int64_t> out(n);

  for (size_t i = 1; i < n; ++i)
    tmp[i] += tmp[i - 1];

  out[0] = tmp[0];
  for (size_t i = 1; i < n; ++i)
    out[i] = out[i - 1] + tmp[i];

  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  DPCM — first-order differencing
// ═══════════════════════════════════════════════════════════════════════════
//
//  Forward:  residuals[0] = samples[0]
//            residuals[i] = samples[i] - samples[i-1]   for i ≥ 1
//
//  Inverse:  values[0]   = residuals[0]
//            values[i]   = values[i-1] + residuals[i]   for i ≥ 1
//            (i.e. inclusive prefix sum / cumulative sum)
//
//  Works for any linearly correlated integer signal.  The first element is
//  stored raw so the inverse can reconstruct without any side-channel data.
//
std::vector<uint8_t> dpcm_encode(std::span<const int64_t> s) {
  size_t n = s.size();
  std::vector<int64_t> d1(n);

  // First element stored as-is
  d1[0] = s[0];

  // First-order differences
  for (size_t i = 1; i < n; ++i)
    d1[i] = s[i] - s[i - 1];

  std::vector<uint8_t> out(n * sizeof(int64_t));
  std::memcpy(out.data(), d1.data(), out.size());
  return out;
}
std::vector<int64_t> dpcm_decode(std::span<const uint8_t> data, size_t n) {
  auto *d1 = reinterpret_cast<const int64_t *>(data.data());

  std::vector<int64_t> out(n);

  // Reconstruct via cumulative sum
  out[0] = d1[0];
  for (size_t i = 1; i < n; ++i)
    out[i] = out[i - 1] + d1[i];

  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
//  XOR encoding — Gorilla-style consecutive XOR on raw bit patterns
// ═══════════════════════════════════════════════════════════════════════════
//
//  Forward:  xor[0] = reinterpret_as_uint64(samples[0])
//            xor[i] = bits(samples[i]) ^ bits(samples[i-1])   for i ≥ 1
//
//  Inverse:  prefix XOR: bits[i] ^= bits[i-1], then reinterpret as int64
//
//  Unlike DPCM (which works on the numeric value), XOR operates on the raw
//  IEEE-754 / two's-complement bit pattern.  Consecutive ECG samples with
//  similar magnitudes share their high-order bits, leaving many leading zeros
//  in the XOR residuals — which Zstd then encodes efficiently.
//
//  The output type is uint64_t (not int64_t) because XOR residuals are not
//  meaningful as signed integers; they are bit differences.
inline std::vector<uint8_t> xor_encode(std::span<const int64_t> samples) {
  const size_t n = samples.size();

  std::vector<uint64_t> tmp(n);
  const uint64_t *src = reinterpret_cast<const uint64_t *>(samples.data());

  tmp[0] = src[0];
  for (size_t i = 1; i < n; ++i)
    tmp[i] = src[i] ^ src[i - 1];

  std::vector<uint8_t> out(n * sizeof(uint64_t));
  std::memcpy(out.data(), tmp.data(), out.size());

  return out;
}

inline std::vector<int64_t> xor_decode(std::span<const uint8_t> data,
                                       size_t n) {
  const uint64_t *encoded = reinterpret_cast<const uint64_t *>(data.data());

  std::vector<uint64_t> out(n);

  out[0] = encoded[0];
  for (size_t i = 1; i < n; ++i)
    out[i] = encoded[i] ^ out[i - 1];

  return std::vector<int64_t>(reinterpret_cast<int64_t *>(out.data()),
                              reinterpret_cast<int64_t *>(out.data()) + n);
}

// ═══════════════════════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════════════════════
//
// Function to get which encode/decode functions to use based on the selected
// mode.
//
struct Transform {
  std::vector<uint8_t> (*encode)(std::span<const int64_t>);
  std::vector<int64_t> (*decode)(std::span<const uint8_t>, size_t n);
};

Transform get_transform(Mode m) {
  switch (m) {
  case Mode::RAW:
    return {raw_encode, raw_decode};
  case Mode::BYTE_SHUFFLE:
    return {shuffle_encode, shuffle_decode};
  case Mode::DOD:
    return {dod_encode, dod_decode};
  case Mode::DPCM:
    return {dpcm_encode, dpcm_decode};
  case Mode::XOR:
    return {xor_encode, xor_decode};
  default:
    return {raw_encode, raw_decode};
  }
}

} // namespace preprocess
