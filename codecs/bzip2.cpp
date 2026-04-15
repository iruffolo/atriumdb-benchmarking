#pragma once
// bzip2.cpp
// ==============
// AtriumDB baseline codec: DPCM pre-filter + BZip2 level 9.
//

#include "codec.hpp"
#include "preprocess.hpp"
#include <bzlib.h>

namespace bzip2 {

static constexpr int BLOCK_SIZE = 9; // 1-9; matches Python compresslevel=9
static constexpr int VERBOSITY = 0;
static constexpr int WORK_FACTOR = 30;

CompressResult compress(std::span<const int16_t> samples,
                        preprocess::Mode mode) {

  auto tform = preprocess::get_transform(mode);

  Timer t;
  t.start();
  auto buffer = tform.encode(samples);
  double encode_ms = t.elapsed_ms();

  double H = preprocess::shannon_entropy_int16(buffer);

  size_t buffer_bytes = buffer.size() * sizeof(int16_t);

  unsigned int bz2_out_len =
      static_cast<unsigned int>(buffer_bytes * 101 / 100 + 600);
  std::vector<uint8_t> out(bz2_out_len);

  Timer t2;
  t2.start();
  int rc = BZ2_bzBuffToBuffCompress(reinterpret_cast<char *>(out.data()),
                                    &bz2_out_len,
                                    reinterpret_cast<char *>(buffer.data()),
                                    static_cast<unsigned int>(buffer_bytes),
                                    BLOCK_SIZE, VERBOSITY, WORK_FACTOR);

  double compress_ms = t2.elapsed_ms();

  if (rc != BZ_OK)
    throw std::runtime_error("BZ2_bzBuffToBuffCompress failed, code=" +
                             std::to_string(rc));

  out.resize(bz2_out_len);
  return {std::move(out), compress_ms, encode_ms, H};
}

DecompressResult decompress(std::span<const uint8_t> data, size_t n_samples,
                            preprocess::Mode mode) {

  auto tform = preprocess::get_transform(mode);

  const size_t bytes = n_samples * sizeof(int16_t);
  std::vector<int16_t> buffer(n_samples);

  unsigned int out_len = static_cast<unsigned int>(bytes);

  Timer t;
  t.start();

  int rc = BZ2_bzBuffToBuffDecompress(
      reinterpret_cast<char *>(buffer.data()), &out_len,
      const_cast<char *>(reinterpret_cast<const char *>(data.data())),
      static_cast<unsigned int>(data.size()), 0, VERBOSITY);

  double decompress_ms = t.elapsed_ms();

  if (rc != BZ_OK)
    throw std::runtime_error("BZ2_bzBuffToBuffDecompress failed, code=" +
                             std::to_string(rc));

  Timer t2;
  t2.start();
  auto values = tform.decode(buffer, n_samples);
  double decode_ms = t2.elapsed_ms();

  return {std::move(values), decompress_ms, decode_ms};
}

} // namespace bzip2
