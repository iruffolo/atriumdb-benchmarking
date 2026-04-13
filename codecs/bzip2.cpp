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

CompressResult compress(std::span<const int64_t> samples,
                        preprocess::Mode mode) {

  auto tform = preprocess::get_transform(mode);

  auto buffer = tform.encode(samples);

  unsigned int bz2_out_len =
      static_cast<unsigned int>(buffer.size() * 101 / 100 + 600);
  std::vector<uint8_t> out(bz2_out_len);

  Timer t;
  t.start();

  int rc = BZ2_bzBuffToBuffCompress(reinterpret_cast<char *>(out.data()),
                                    &bz2_out_len,
                                    reinterpret_cast<char *>(buffer.data()),
                                    static_cast<unsigned int>(buffer.size()),
                                    BLOCK_SIZE, VERBOSITY, WORK_FACTOR);

  double ms = t.elapsed_ms();

  if (rc != BZ_OK)
    throw std::runtime_error("BZ2_bzBuffToBuffCompress failed, code=" +
                             std::to_string(rc));

  out.resize(bz2_out_len);
  return {std::move(out), ms};
}

DecompressResult decompress(std::span<const uint8_t> data, size_t n_samples,
                            preprocess::Mode mode) {

  auto tform = preprocess::get_transform(mode);

  const size_t bytes = n_samples * sizeof(int64_t);
  std::vector<uint8_t> buffer(bytes);

  Timer t;
  t.start();

  unsigned int out_len = static_cast<unsigned int>(bytes);

  int rc = BZ2_bzBuffToBuffDecompress(
      reinterpret_cast<char *>(buffer.data()), &out_len,
      const_cast<char *>(reinterpret_cast<const char *>(data.data())),
      static_cast<unsigned int>(data.size()), 0, VERBOSITY);

  if (rc != BZ_OK)
    throw std::runtime_error("BZ2_bzBuffToBuffDecompress failed, code=" +
                             std::to_string(rc));

  auto values = tform.decode(buffer, n_samples);

  double ms = t.elapsed_ms();
  return {std::move(values), ms};
}

} // namespace bzip2
