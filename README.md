# AtriumDB Compression Benchmark

Benchmarks twenty lossless compression pipelines for integer-valued physiological waveforms stored in AtriumDB, using the MIT-BIH Arrhythmia Database as the evaluation corpus.

## Dependencies

- AtriumDB SDK
- pybind11
- zstd, liblzma, bzip2 (system libraries)
- wfdb (Python, for MIT-BIH ingestion)


## Build

```bash
mkdir build && cd build
cmake ..
make
```

## Usage

**Step 1 — Ingest MIT-BIH into AtriumDB:**
```bash
mkdir database && cd scipts
python build_mitdb_dataset.py
```

**Step 2 — Run the benchmark:**
```bash
python benchmark.py
```

Results are printed to stdout as a table of compression ratio, bits per value, Shannon entropy, coding efficiency, and compression/decompression throughput for all twenty pipelines.

## Structure

```
codecs/
  codec.hpp         # Shared CompressResult/DecompressResult structs and Timer
  preprocess.hpp    # Transform implementations (DPCM, DOD, XOR, byte-shuffle)
  zstd.cpp          # Zstandard codec
  lzma.cpp          # LZMA codec
  bzip2.cpp         # BZip2 codec
  golomb.cpp        # Adaptive Rice-Golomb codec
  bindings.cpp      # pybind11 bindings — single Python-facing translation unit
```
