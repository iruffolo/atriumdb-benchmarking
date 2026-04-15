#pragma once
// golomb.cpp
// ==========
// Adaptive Rice coding (Golomb coding restricted to M = 2^k) applied to
// DPCM residuals of int64 physiological signal samples.
//
// Background
// ----------
// Golomb coding is provably optimal for geometric distributions — i.e. where
// the probability of a value n falls as p^n for some 0 < p < 1.  DPCM
// residuals from slowly-varying physiological waveforms (ECG, ABP, SpO2)
// closely follow this distribution: most differences are small (±1–10 ADU),
// with exponentially decreasing probability of larger jumps.
//
// Rice coding is the special case M = 2^k, which replaces the general Golomb
// division by a fast bit-shift.  It is used in FLAC audio, CCSDS space data
// compression, and many biomedical signal codecs.
//
// Encoding a non-negative integer n with parameter k:
//   quotient  q = n >> k          (n / 2^k)
//   remainder r = n & (2^k - 1)   (n mod 2^k)
//   bitstream: q ones, one zero, then k bits of r (LSB first)
//
// Signed integers are mapped to non-negative via zigzag encoding before
// coding, and unmapped on decode — same scheme used by Protocol Buffers:
//   zigzag(x) = (x << 1) ^ (x >> 63)   maps  0→0, -1→1, 1→2, -2→3, …
//   unzigzag(z) = (z >> 1) ^ -(z & 1)
//
// Adaptive k selection
// --------------------
// The optimal k for a geometric distribution with mean μ is:
//   k = max(0, round(log2(log(2) * μ)))
// We estimate μ from the mean absolute value of the residuals in a training
// pass over the first min(1024, n) samples, then use that k for the entire
// segment.  The chosen k is stored as a single byte header so the decoder
// can reconstruct without side-channel information.
//
// Bit-packing
// -----------
// Bits are packed into a std::vector<uint8_t> MSB-first within each byte.
// A 64-bit accumulator is used internally to minimise per-bit branching.
// The stream is zero-padded to a byte boundary at the end.
//
// The compressed format is:
//   [1 byte : k value]  [n_samples as uint64_t little-endian]  [packed bits…]
//
// Timer wraps:
//   compress:   DPCM pass + zigzag + Rice encode
//   decompress: Rice decode + unzigzag + cumsum

#include "codec.hpp"
#include "preprocess.hpp"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace golomb {

// Zigzag map
static inline uint16_t zigzag_encode(int16_t x) {
  return static_cast<uint16_t>((static_cast<uint16_t>(x) << 1) ^
                               static_cast<uint16_t>(x >> 15));
}

static inline int16_t zigzag_decode(uint16_t z) {
  return static_cast<int16_t>((z >> 1) ^ -static_cast<int16_t>(z & 1u));
}

static constexpr size_t TRAIN_N = 1024;
static constexpr int MAX_K = 30; // k=30 → quotients are almost never >0

static int estimate_k(std::span<const int16_t> residuals) {
  const size_t train = std::min(residuals.size(), TRAIN_N);
  double sum_abs = 0.0;
  for (size_t i = 0; i < train; ++i)
    sum_abs += static_cast<double>(std::abs(residuals[i]));
  double mu = (2.0 * sum_abs / static_cast<double>(train)) + 0.5;
  if (mu < 1.0)
    mu = 1.0;
  int k = static_cast<int>(std::round(std::log2(std::log(2.0) * mu)));
  return std::max(0, std::min(k, MAX_K));
}

struct BitWriter {
  std::vector<uint8_t> &out;
  uint64_t buf = 0;
  int bits = 0;

  explicit BitWriter(std::vector<uint8_t> &out) : out(out) {}

  void write(uint64_t val, int n) {
    buf = (buf << n) | (val & ((uint64_t(1) << n) - 1));
    bits += n;
    while (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<uint8_t>(buf >> bits));
    }
  }

  void write_zero() { write(0, 1); }

  void flush() {
    if (bits > 0) {
      out.push_back(static_cast<uint8_t>(buf << (8 - bits)));
      bits = 0;
      buf = 0;
    }
  }
};

struct BitReader {
  const uint8_t *src;
  size_t byte_len;
  size_t byte_pos = 0;
  uint64_t buf = 0;
  int bits = 0;

  BitReader(const uint8_t *src, size_t len) : src(src), byte_len(len) {}

  void refill() {
    while (bits <= 56 && byte_pos < byte_len) {
      buf = (buf << 8) | src[byte_pos++];
      bits += 8;
    }
  }

  uint64_t read(int n) {
    if (bits < n)
      refill();
    bits -= n;
    return (buf >> bits) & ((uint64_t(1) << n) - 1);
  }

  uint64_t read_unary() {
    uint64_t q = 0;
    for (;;) {
      if (bits < 1)
        refill();
      if (bits == 0)
        throw std::runtime_error("golomb::decompress: truncated bitstream");
      --bits;
      if (((buf >> bits) & 1) == 0)
        break;
      ++q;
    }
    return q;
  }
};

static void rice_encode_value(BitWriter &bw, uint64_t z, int k) {
  const uint64_t q = z >> k;
  const uint64_t r = z & ((uint64_t(1) << k) - 1);
  uint64_t remaining = q;
  while (remaining > 0) {
    int batch = static_cast<int>(std::min(remaining, uint64_t(57)));
    uint64_t mask = (batch == 64) ? ~uint64_t(0) : ((uint64_t(1) << batch) - 1);
    bw.write(mask, batch);
    remaining -= batch;
  }
  bw.write_zero();
  if (k > 0)
    bw.write(r, k);
}

CompressResult compress(std::span<const int16_t> samples,
                        preprocess::Mode mode) {
  const size_t n = samples.size();
  auto tform = preprocess::get_transform(mode);

  Timer t;
  t.start();
  std::vector<int16_t> buffer = tform.encode(samples);
  double encode_ms = t.elapsed_ms();

  double H = preprocess::shannon_entropy_int16(buffer);

  const int k = estimate_k(buffer);
  std::cout << k << std::endl;

  std::vector<uint8_t> out;
  out.reserve(n * 3);

  // Header: k (1 byte) + n_samples (8 bytes, little-endian)
  out.push_back(static_cast<uint8_t>(k));
  const uint64_t n64 = static_cast<uint64_t>(n);
  for (int b = 0; b < 8; ++b)
    out.push_back(static_cast<uint8_t>((n64 >> (b * 8)) & 0xFF));

  BitWriter bw(out);
  for (size_t i = 0; i < n; ++i)
    rice_encode_value(bw, zigzag_encode(buffer[i]), k);
  bw.flush();

  double compress_ms = t.elapsed_ms() - encode_ms;
  return {std::move(out), compress_ms, encode_ms, H};
}

DecompressResult decompress(std::span<const uint8_t> data, size_t /*n_samples*/,
                            preprocess::Mode mode) {

  const int k = static_cast<int>(data[0]);
  uint64_t n64 = 0;
  for (int b = 0; b < 8; ++b)
    n64 |= static_cast<uint64_t>(data[1 + b]) << (b * 8);
  const size_t n = static_cast<size_t>(n64);

  if (k < 0 || k > MAX_K)
    throw std::runtime_error("golomb::decompress: invalid k=" +
                             std::to_string(k));

  auto tform = preprocess::get_transform(mode);

  Timer t;
  t.start();

  std::vector<int16_t> buffer(n);
  BitReader br(data.data() + 9, data.size() - 9);
  for (size_t i = 0; i < n; ++i) {
    uint64_t q = br.read_unary();
    uint64_t r = (k > 0) ? br.read(k) : 0;
    buffer[i] = zigzag_decode(static_cast<uint16_t>((q << k) | r));
  }

  double decompress_ms = t.elapsed_ms();

  Timer t2;
  t2.start();
  auto values = tform.decode(buffer, n);
  double decode_ms = t2.elapsed_ms();

  return {std::move(values), decompress_ms, decode_ms};
}

} // namespace golomb
