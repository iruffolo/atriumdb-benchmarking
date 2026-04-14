// zstd.cpp
// ============
//

#pragma once
#include "codec.hpp"
#include "preprocess.hpp"
#include <span>
#include <zstd.h>

namespace zstd {

static constexpr int ZSTD_LEVEL = 3;

CompressResult compress(std::span<const int64_t> samples,
                        preprocess::Mode mode) {

  auto tform = preprocess::get_transform(mode);

  Timer t;
  t.start();
  auto buffer = tform.encode(samples);
  double encode_ms = t.elapsed_ms();

  double H = preprocess::shannon_entropy(buffer);

  size_t bound = ZSTD_compressBound(buffer.size());
  std::vector<uint8_t> out(bound);

  Timer t2;
  t2.start();
  size_t compressed_size = ZSTD_compress(out.data(), bound, buffer.data(),
                                         buffer.size(), ZSTD_LEVEL);
  double compress_ms = t2.elapsed_ms();

  if (ZSTD_isError(compressed_size))
    throw std::runtime_error(std::string("ZSTD_compress failed: ") +
                             ZSTD_getErrorName(compressed_size));

  out.resize(compressed_size);
  return {std::move(out), compress_ms, encode_ms, H};
}

DecompressResult decompress(std::span<const uint8_t> data, size_t n_samples,
                            preprocess::Mode mode) {

  auto tform = preprocess::get_transform(mode);

  size_t bytes = n_samples * sizeof(int64_t);
  std::vector<uint8_t> buffer(bytes);

  Timer t;
  t.start();

  size_t result =
      ZSTD_decompress(buffer.data(), bytes, data.data(), data.size());

  double decompress_ms = t.elapsed_ms();

  if (ZSTD_isError(result))
    throw std::runtime_error(std::string("ZSTD_decompress failed: ") +
                             ZSTD_getErrorName(result));

  Timer t2;
  t2.start();
  auto values = tform.decode(buffer, n_samples);
  double decode_ms = t2.elapsed_ms();

  return {std::move(values), decompress_ms, decode_ms};
}

} // namespace zstd
