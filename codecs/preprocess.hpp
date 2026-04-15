#pragma once

// preprocess.hpp
// ==============
// Preprocessing transforms for time-series compression.
//
// Provides reversible transforms (DPCM, delta-of-delta, XOR, byte shuffle,
// raw) that convert int16_t sample sequences into byte buffers suitable for
// compression. Each transform defines an encode/decode pair and can be
// combined with any backend compressor (Zstd, Bzip2, LZMA, etc.).
//

#include <cmath>
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

struct Transform {
  std::vector<int16_t> (*encode)(std::span<const int16_t>);
  std::vector<int16_t> (*decode)(std::span<const int16_t>, size_t n);
};

// ═══════════════════════════════════════════════════════════════════════════
//  Shannon Entropy calculation
// ═══════════════════════════════════════════════════════════════════════════
//
double shannon_entropy(std::span<const uint8_t> data) {
  if (data.empty())
    return 0.0;

  // Histogram (256 possible byte values)
  std::array<size_t, 256> counts{};

  for (uint8_t b : data)
    counts[b]++;

  const double N = static_cast<double>(data.size());

  double H = 0.0;

  for (size_t c : counts) {
    if (c == 0)
      continue;

    double p = c / N;
    H -= p * std::log2(p);
  }

  return H; // bits per byte
}

double shannon_entropy_int16(std::span<const int16_t> data) {
  if (data.empty())
    return 0.0;

  // 2^16 possible values
  std::array<uint32_t, 65536> counts{};

  // Build histogram
  for (int16_t v : data) {
    // Map signed [-32768, 32767] → [0, 65535]
    uint16_t idx = static_cast<uint16_t>(v);
    counts[idx]++;
  }

  const double N = static_cast<double>(data.size());

  double H = 0.0;

  for (uint32_t c : counts) {
    if (c == 0)
      continue;

    double p = c / N;
    H -= p * std::log2(p);
  }

  return H; // bits per value
}

// ═══════════════════════════════════════════════════════════════════════════
//  Raw encode
// ═══════════════════════════════════════════════════════════════════════════
//

inline std::vector<int16_t> raw_encode(std::span<const int16_t> s) {
  return std::vector<int16_t>(s.begin(), s.end());
}

inline std::vector<int16_t> raw_decode(std::span<const int16_t> data,
                                       size_t n) {
  return std::vector<int16_t>(data.begin(), data.begin() + n);
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
//  The output is a flat byte buffer of length n * sizeof(int16_t).
//  ELEM_BYTES is a template parameter so the same code works for int32_t,
//  float, double etc. if needed in the future.
//

inline std::vector<int16_t> shuffle_encode(std::span<const int16_t> s) {
  size_t n = s.size();
  size_t B = sizeof(int16_t);
  std::vector<int16_t> out(n);
  auto *src = reinterpret_cast<const uint8_t *>(s.data());
  auto *dst = reinterpret_cast<uint8_t *>(out.data());
  for (size_t b = 0; b < B; ++b)
    for (size_t i = 0; i < n; ++i)
      dst[b * n + i] = src[i * B + b];
  return out;
}

inline std::vector<int16_t> shuffle_decode(std::span<const int16_t> data,
                                           size_t n) {
  size_t B = sizeof(int16_t);
  std::vector<int16_t> out(n);
  auto *dst = reinterpret_cast<uint8_t *>(out.data());
  auto *src = reinterpret_cast<const uint8_t *>(data.data());
  for (size_t b = 0; b < B; ++b)
    for (size_t i = 0; i < n; ++i)
      dst[i * B + b] = src[b * n + i];
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
inline std::vector<int16_t> dod_encode(std::span<const int16_t> s) {
  size_t n = s.size();
  std::vector<int16_t> d1(n), d2(n);

  d1[0] = s[0];
  for (size_t i = 1; i < n; ++i)
    d1[i] = s[i] - s[i - 1];

  d2[0] = d1[0];
  for (size_t i = 1; i < n; ++i)
    d2[i] = d1[i] - d1[i - 1];

  // std::vector<uint8_t> out(n * sizeof(int16_t));
  // std::memcpy(out.data(), d2.data(), out.size());
  return d2;
}

std::vector<int16_t> dod_decode(std::span<const int16_t> data, size_t n) {
  std::vector<int16_t> tmp(data.begin(), data.begin() + n);
  std::vector<int16_t> out(n);
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

inline std::vector<int16_t> dpcm_encode(std::span<const int16_t> s) {
  size_t n = s.size();
  std::vector<int16_t> d1(n);

  // First element stored as-is
  d1[0] = s[0];

  // First-order differences
  for (size_t i = 1; i < n; ++i)
    d1[i] = s[i] - s[i - 1];

  // std::vector<uint8_t> out(n * sizeof(int16_t));
  // std::memcpy(out.data(), d1.data(), out.size());
  return d1;
}
inline std::vector<int16_t> dpcm_decode(std::span<const int16_t> data,
                                        size_t n) {

  // auto *d1 = reinterpret_cast<const int16_t *>(data.data());
  std::vector<int16_t> out(n);

  // Reconstruct via cumulative sum
  out[0] = data[0];
  for (size_t i = 1; i < n; ++i)
    out[i] = out[i - 1] + data[i];

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
//  The output type is uint16_t (not int16_t) because XOR residuals are not
//  meaningful as signed integers; they are bit differences.
inline std::vector<int16_t> xor_encode(std::span<const int16_t> samples) {
  const size_t n = samples.size();

  std::vector<int16_t> tmp(n);
  const int16_t *src = reinterpret_cast<const int16_t *>(samples.data());

  tmp[0] = src[0];
  for (size_t i = 1; i < n; ++i)
    tmp[i] = src[i] ^ src[i - 1];

  // std::vector<uint8_t> out(n * sizeof(uint16_t));
  // std::memcpy(out.data(), tmp.data(), out.size());
  return tmp;
}

inline std::vector<int16_t> xor_decode(std::span<const int16_t> data,
                                       size_t n) {
  const uint16_t *encoded = reinterpret_cast<const uint16_t *>(data.data());
  std::vector<uint16_t> out(n);
  out[0] = encoded[0];
  for (size_t i = 1; i < n; ++i)
    out[i] = encoded[i] ^ out[i - 1];
  return std::vector<int16_t>(reinterpret_cast<int16_t *>(out.data()),
                              reinterpret_cast<int16_t *>(out.data()) + n);
}

// Helper to get which encode/decode functions to use
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
