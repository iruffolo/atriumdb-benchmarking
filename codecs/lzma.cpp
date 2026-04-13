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

CompressResult compress(std::span<const int64_t> samples,
                        preprocess::Mode mode) {

  auto tform = preprocess::get_transform(mode);

  Timer t;
  t.start();

  std::vector<uint8_t> buffer = tform.encode(samples);

  size_t bound = lzma_stream_buffer_bound(buffer.size());
  std::vector<uint8_t> out(bound);
  size_t out_pos = 0;

  lzma_ret rc =
      lzma_easy_buffer_encode(PRESET, LZMA_CHECK_CRC64, nullptr, buffer.data(),
                              buffer.size(), out.data(), &out_pos, bound);

  double ms = t.elapsed_ms();

  if (rc != LZMA_OK)
    throw std::runtime_error("lzma_easy_buffer_encode failed, code=" +
                             std::to_string(rc));

  out.resize(out_pos);
  return {std::move(out), ms};
}

DecompressResult decompress(std::span<const uint8_t> data, size_t n_samples,
                            preprocess::Mode mode) {
  if (data.empty())
    throw std::runtime_error("lzma::decompress: empty input");

  auto tform = preprocess::get_transform(mode);

  const size_t bytes = n_samples * sizeof(int64_t);
  std::vector<uint8_t> buffer(bytes);

  size_t in_pos = 0, out_pos = 0;

  Timer t;
  t.start();

  lzma_ret rc =
      lzma_stream_buffer_decode(&MEM_LIMIT, 0, nullptr, data.data(), &in_pos,
                                data.size(), buffer.data(), &out_pos, bytes);

  if (rc != LZMA_OK && rc != LZMA_STREAM_END)
    throw std::runtime_error("lzma_stream_buffer_decode failed, code=" +
                             std::to_string(rc));

  auto values = tform.decode(buffer, n_samples);

  double ms = t.elapsed_ms();
  return {std::move(values), ms};
}

} // namespace lzma
