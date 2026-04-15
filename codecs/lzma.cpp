#pragma once
// lzma.cpp
// =============
// DPCM pre-filter + LZMA level 6.  High-ratio reference codec.
//

#include "codec.hpp"
#include "preprocess.hpp"
#include <lzma.h>

namespace lzma {

static constexpr uint32_t PRESET = 6;
static uint64_t MEM_LIMIT = UINT64_MAX;

CompressResult compress(std::span<const int16_t> samples,
                        preprocess::Mode mode) {
  auto tform = preprocess::get_transform(mode);

  Timer t;
  t.start();
  auto buffer = tform.encode(samples);
  double encode_ms = t.elapsed_ms();

  double H = preprocess::shannon_entropy_int16(buffer);

  size_t buffer_bytes = buffer.size() * sizeof(int16_t);
  size_t bound = lzma_stream_buffer_bound(buffer_bytes);
  std::vector<uint8_t> out(bound);
  size_t out_pos = 0;

  Timer t2;
  t2.start();
  lzma_ret rc =
      lzma_easy_buffer_encode(PRESET, LZMA_CHECK_CRC64, nullptr,
                              reinterpret_cast<const uint8_t *>(buffer.data()),
                              buffer_bytes, out.data(), &out_pos, bound);
  double compress_ms = t2.elapsed_ms();

  if (rc != LZMA_OK)
    throw std::runtime_error("lzma_easy_buffer_encode failed, code=" +
                             std::to_string(rc));

  out.resize(out_pos);
  return {std::move(out), compress_ms, encode_ms, H};
}

DecompressResult decompress(std::span<const uint8_t> data, size_t n_samples,
                            preprocess::Mode mode) {

  auto tform = preprocess::get_transform(mode);
  const size_t bytes = n_samples * sizeof(int16_t);
  std::vector<int16_t> buffer(n_samples);
  size_t in_pos = 0, out_pos = 0;

  Timer t;
  t.start();
  lzma_ret rc = lzma_stream_buffer_decode(
      &MEM_LIMIT, 0, nullptr, data.data(), &in_pos, data.size(),
      reinterpret_cast<uint8_t *>(buffer.data()), &out_pos, bytes);
  double decompress_ms = t.elapsed_ms();

  if (rc != LZMA_OK && rc != LZMA_STREAM_END)
    throw std::runtime_error("lzma_stream_buffer_decode failed, code=" +
                             std::to_string(rc));

  Timer t2;
  t2.start();
  auto values = tform.decode(buffer, n_samples);
  double decode_ms = t2.elapsed_ms();

  return {std::move(values), decompress_ms, decode_ms};
}

} // namespace lzma
