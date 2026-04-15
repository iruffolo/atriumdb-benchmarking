"""
AtriumDB Compression Benchmark Suite
=====================================
Tests AtriumDB's native DPCM+BZip2 compression against alternative methods
using the MIT-BIH Arrhythmia Database ingested by build_mitdb_dataset.py.

Requirements
------------
    pip install atriumdb wfdb numpy zstandard tqdm

Output
------
- Per-segment comparison table printed to stdout
- Aggregate summary with % improvement vs AtriumDB baseline
- compression_results.json saved alongside this script
"""

import bz2
import json
import lzma
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np
from atriumdb import AtriumSDK

import build.compression_codecs as _codecs

# ══════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════
RANDOM_SEED = 42


def load_segments_from_atriumdb(
    dataset_path: str,
    n_segments,  # int or the string 'all'
    duration_s: int,
) -> List[dict]:
    """
    Open the AtriumDB dataset built by build_mitdb_dataset.py and pull signal
    segments for benchmarking.

    MIT-BIH layout inside AtriumDB:
      • Each record (100, 101, …, 234) is a separate *device*
      • Each ECG lead (MLII, V1, V2, …) is a *measure*
      • All segments are anchored at t=0 ns; duration ≈ 30 min (650,000 samples)
      • Sampling rate: 360 Hz  →  freq_nhz = 360_000_000_000

    Returns a list of dicts:
        {
          'label':       str,          # "record/channel" e.g. "100/MLII"
          'measure_tag': str,
          'device_tag':  str,
          'freq_hz':     float,
          'values':      np.ndarray,   # int64 digital ECG samples
          'times_ns':    np.ndarray,   # nanosecond timestamps
        }
    """

    sdk = AtriumSDK(dataset_location=dataset_path)

    all_measures = sdk.get_all_measures()  # {measure_id: {tag, freq_nhz, units, ...}}
    all_devices = sdk.get_all_devices()  # {device_id:  {tag, ...}}
    print(
        f"[INFO] Dataset: {len(all_measures)} measure(s) across "
        f"{len(all_devices)} device(s) (records)."
    )

    # Build all valid (measure_id, device_id) pairs that have data
    rng = np.random.default_rng(RANDOM_SEED)
    pairs = [(mid, did) for mid in all_measures for did in all_devices]
    # Shuffle for variety across signal types and records
    idx = rng.permutation(len(pairs))
    pairs = [pairs[i] for i in idx]

    want = len(pairs) if n_segments == "all" else int(n_segments)
    print(
        f"[INFO] Sampling up to {want} (measure, device) pairs, "
        f"{duration_s}s each.\n"
    )

    segments = []

    for measure_id, device_id in pairs:
        if len(segments) >= want:
            break

        measure_info = all_measures[measure_id]
        device_info = all_devices[device_id]

        freq_nhz = measure_info.get("freq_nhz", 0)
        if not freq_nhz:
            continue

        freq_hz = freq_nhz / 1e9

        # Check this pair actually has data (not every lead exists in every record)
        try:
            interval_arr = sdk.get_interval_array(
                measure_id=measure_id, device_id=device_id
            )
        except Exception:
            continue
        if interval_arr is None or len(interval_arr) == 0:
            continue

        # Pull `duration_s` seconds from the start of the record
        start_ns = int(interval_arr[0, 0])
        end_ns = start_ns + int(duration_s * 1e9)
        end_ns = min(end_ns, int(interval_arr[-1, 1]))

        _, times_ns, values = sdk.get_data(
            measure_id=measure_id,
            device_id=device_id,
            start_time_n=start_ns,
            end_time_n=end_ns,
        )

        if values is None or len(values) < 100:
            continue

        measure_tag = measure_info.get("tag", f"m{measure_id}")
        device_tag = device_info.get("tag", f"d{device_id}")
        label = f"{device_tag}/{measure_tag}"

        segments.append(
            {
                "label": label,
                "measure_tag": measure_tag,
                "device_tag": device_tag,
                "freq_hz": freq_hz,
                "values": np.asarray(values, dtype=np.int64),
                "times_ns": np.asarray(times_ns),
            }
        )
        print(
            f"  ✓  {label:<18}  {freq_hz:.0f} Hz  "
            f"{len(values):>8,} samples  "
            f"({len(values)/freq_hz:.0f}s)"
        )

    print(f"\n  Loaded {len(segments)} segment(s).\n")
    return segments


# ══════════════════════════════════════════════
def measure_atriumdb_ondisk(dataset_path: str) -> Optional[int]:
    """
    Sum up the bytes used by all .tsc (compressed) data files in the dataset.
    Returns total bytes, or None if the path cannot be found.
    """
    p = Path(dataset_path)
    tsc_files = list(p.rglob("*.tsc"))
    if not tsc_files:
        return None
    return sum(f.stat().st_size for f in tsc_files)


# ══════════════════════════════════════════════
#  BENCHMARK HARNESS
# ══════════════════════════════════════════════


@dataclass
class CompressorSpec:
    name: str
    compress_fn: Callable[[np.ndarray], bytes]
    decompress_fn: Callable[[bytes, type, int], np.ndarray]
    note: str = ""


@dataclass
class BenchResult:
    compressor: str
    note: str
    label: str  # "record/channel" e.g. "100/MLII"
    measure_tag: str
    device_tag: str
    n_samples: int
    raw_bytes: int
    compressed_bytes: int
    compression_ratio: float
    bits_per_value: float
    shannon_entropy: float
    # ── internal C++ timing (std::chrono, algorithmic work only) ────────────
    # None for pure-Python codecs.
    cpp_compress_ms: float = None
    cpp_decompress_ms: float = None
    cpp_encode_ms: float = None
    cpp_decode_ms: float = None
    roundtrip_ok: bool = False
    error: str = ""

    @property
    def compress_overhead_ms(self) -> Optional[float]:
        """Wall-clock minus internal C++ time = Python↔C++ call overhead."""
        if self.cpp_compress_ms is None:
            return None
        return max(0.0, self.compress_ms - self.cpp_compress_ms)

    @property
    def decompress_overhead_ms(self) -> Optional[float]:
        if self.cpp_decompress_ms is None:
            return None
        return max(0.0, self.decompress_ms - self.cpp_decompress_ms)


def _infer_decode_dtype(values: np.ndarray):
    if np.issubdtype(values.dtype, np.floating):
        return np.float64
    return np.int64


def benchmark_one(
    spec: CompressorSpec,
    seg: dict,
    n_trials: int = 3,
) -> BenchResult:
    """
    Run compress + decompress n_trials times, keep the best (fastest) run.

    For C++ codecs:
      • compress_fn returns (bytes, cpp_compress_ms)
      • decompress_fn returns (np.ndarray, cpp_decompress_ms)
      • We record both the Python wall-clock time and the internal C++ time.
      • BenchResult.compress_overhead_ms = wall − cpp  (pure marshalling cost)

    """
    values = seg["values"]
    n_samples = len(values)
    raw_bytes = values.nbytes
    dtype = _infer_decode_dtype(values)
    label = seg.get("label", f"{seg['device_tag']}/{seg['measure_tag']}")

    def _make_error(msg: str) -> BenchResult:
        return BenchResult(
            compressor=spec.name,
            note=spec.note,
            label=label,
            measure_tag=seg["measure_tag"],
            device_tag=seg["device_tag"],
            n_samples=n_samples,
            raw_bytes=raw_bytes,
            compressed_bytes=0,
            compression_ratio=0.0,
            compress_ms=0.0,
            decompress_ms=0.0,
            cpp_compress_ms=None,
            cpp_decompress_ms=None,
            roundtrip_ok=False,
            error=msg,
        )

    # ── compression ──────────────────────────────────────────────────────────
    compressed = None
    best_cpp_comp_ms = float("inf")
    best_cpp_encode_ms = float("inf")

    try:
        for _ in range(n_trials):
            result = spec.compress_fn(values)
            raw_out, cpp_ms, enc_ms, shannon = result

            compressed = bytes(raw_out)  # one copy into Python bytes

            if cpp_ms < best_cpp_comp_ms:
                best_cpp_comp_ms = cpp_ms
            if enc_ms < best_cpp_encode_ms:
                best_cpp_encode_ms = enc_ms

    except Exception as e:
        return _make_error(str(e))

    compressed_bytes = len(compressed)
    compression_ratio = raw_bytes / compressed_bytes if compressed_bytes else 0.0

    bits_per_value = (compressed_bytes * 8) / n_samples if compressed_bytes else 0.0

    # ── decompression ─────────────────────────────────────────────────────────
    best_cpp_decomp_ms = float("inf")
    best_cpp_decode_ms = float("inf")
    roundtrip_ok = False
    error_msg = ""

    try:
        for _ in range(n_trials):
            result = spec.decompress_fn(compressed, dtype, n_samples)

            recovered, cpp_ms, dec_ms = result

            if cpp_ms < best_cpp_decomp_ms:
                best_cpp_decomp_ms = cpp_ms
            if dec_ms < best_cpp_decode_ms:
                best_cpp_decode_ms = dec_ms

        # Lossless round-trip check
        orig = values.astype(dtype)
        recovered_arr = np.asarray(recovered)
        if np.issubdtype(dtype, np.floating):
            roundtrip_ok = bool(
                np.allclose(orig, recovered_arr[:n_samples], rtol=0, atol=0)
            )
        else:
            roundtrip_ok = bool(np.array_equal(orig, recovered_arr[:n_samples]))

    except Exception as e:
        error_msg = str(e)

    return BenchResult(
        compressor=spec.name,
        note=spec.note,
        label=label,
        measure_tag=seg["measure_tag"],
        device_tag=seg["device_tag"],
        n_samples=n_samples,
        raw_bytes=raw_bytes,
        compressed_bytes=compressed_bytes,
        compression_ratio=round(compression_ratio, 3),
        bits_per_value=round(bits_per_value, 3),
        shannon_entropy=round(shannon, 3),
        cpp_compress_ms=(
            round(best_cpp_comp_ms, 3) if best_cpp_comp_ms is not None else None
        ),
        cpp_decompress_ms=(
            round(best_cpp_decomp_ms, 3) if best_cpp_decomp_ms is not None else None
        ),
        cpp_encode_ms=best_cpp_encode_ms,
        cpp_decode_ms=best_cpp_decode_ms,
        roundtrip_ok=roundtrip_ok,
        error=error_msg,
    )


# ══════════════════════════════════════════════
#  AGGREGATE SUMMARY
# ══════════════════════════════════════════════


def print_summary(results: List[BenchResult]):
    ok_results = [r for r in results if r.roundtrip_ok]
    if not ok_results:
        print("[WARN] No successful round-trips to summarise.")
        return

    has_cpp = any(r.cpp_compress_ms is not None for r in ok_results)
    compressors = list(dict.fromkeys(r.compressor for r in ok_results))

    # ── Per-codec aggregate stats ─────────────────────────────────────────────
    rows = []
    for cname in compressors:
        cr = [r for r in ok_results if r.compressor == cname]

        avg_ratio = float(np.mean([r.compression_ratio for r in cr]))

        avg_bits_per_val = float(np.mean([r.bits_per_value for r in cr]))

        avg_shannon = float(np.mean([r.shannon_entropy for r in cr]))

        # C++ internal MB/s — None for pure-Python codecs
        cpp_rows = [r for r in cr if r.cpp_compress_ms is not None]
        avg_cpp_comp_mbs = (
            float(
                np.mean(
                    [
                        r.raw_bytes / max(r.cpp_compress_ms, 0.001) / 1000
                        for r in cpp_rows
                    ]
                )
            )
            if cpp_rows
            else None
        )
        avg_cpp_decomp_mbs = (
            float(
                np.mean(
                    [
                        r.raw_bytes / max(r.cpp_decompress_ms, 0.001) / 1000
                        for r in cpp_rows
                    ]
                )
            )
            if cpp_rows
            else None
        )
        avg_cpp_enc_mbs = (
            float(
                np.mean(
                    [r.raw_bytes / max(r.cpp_encode_ms, 0.001) / 1000 for r in cpp_rows]
                )
            )
            if cpp_rows
            else None
        )
        avg_cpp_decode_mbs = (
            float(
                np.mean(
                    [r.raw_bytes / max(r.cpp_decode_ms, 0.001) / 1000 for r in cpp_rows]
                )
            )
            if cpp_rows
            else None
        )

        rows.append(
            {
                "name": cname,
                "note": cr[0].note,
                "avg_ratio": avg_ratio,
                "avg_bits_per_val": avg_bits_per_val,
                "avg_shannon": avg_shannon,
                "cpp_comp_mbs": avg_cpp_comp_mbs,
                "cpp_decomp_mbs": avg_cpp_decomp_mbs,
                "cpp_enc_mbs": avg_cpp_enc_mbs,
                "cpp_dec_mbs": avg_cpp_decode_mbs,
            }
        )

    # ── Print table ───────────────────────────────────────────────────────────
    print(f"\n{'═'*100}")
    print("  AGGREGATE SUMMARY  (mean across all segments, lossless runs only)")
    print(f"{'═'*100}")

    if has_cpp:
        print(
            f"  {'Algorithm':<28}  {'Avg Ratio':>14}  {'Avg Bits per Val':>20}"
            f"{'Avg Shannon':>20}  {'C++ comp MB/s':>16}  {'C++ dec MB/s':>16}  "
            f"{'Note':>10}"
        )
        print(f"  {'─'*28}  {'─'*14}  {'─'*20}  {'─'*20}  {'─'*16}  {'─'*16}  {'─'*20}")

        for r in sorted(rows, key=lambda x: -x["avg_ratio"]):
            cpp_c = (
                f"{r['cpp_comp_mbs']:>13.0f}" if r["cpp_comp_mbs"] else "            -"
            )
            cpp_d = (
                f"{r['cpp_decomp_mbs']:>12.0f}"
                if r["cpp_decomp_mbs"]
                else "           -"
            )
            print(
                f"  {r['name']:<28}  {r['avg_ratio']:>14f}x  {r['avg_bits_per_val']:>20f}"
                f"  {r['avg_shannon']:>20f} {cpp_c:<16}  {cpp_d:<16}  {r['note']}"
            )

    # ── Baseline delta table ──────────────────────────────────────────────────
    baseline = next((r for r in rows if "DPCM+BZip2" in r["name"]), None)
    if baseline:
        print(f"\n  Baseline : {baseline['name']}")
        print(
            f"  {'vs':<36}  {'comp ratio Δ':>16}  {'bpv ratio Δ':>16} {'C++ comp Δ':>16}"
        )
        print(f"  {'─'*36}  {'─'*16}  {'─'*16}  {'─'*16} ")

        for r in sorted(rows, key=lambda x: -x["avg_ratio"]):
            if r["name"] == baseline["name"]:
                continue

            def _pct(a, b):
                return f"{((a - b) / max(b, 0.001)) * 100:+.1f}%"

            ratio_d = _pct(r["avg_ratio"], baseline["avg_ratio"])
            ratio_bpv = _pct(r["avg_bits_per_val"], baseline["avg_bits_per_val"])
            cppc_d = _pct(r["cpp_comp_mbs"], baseline["cpp_comp_mbs"])

            print(f"  {r['name']:<36}  {ratio_d:>16}  {ratio_bpv:>16}  {cppc_d:>16}  ")

    print()


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         AtriumDB Compression Benchmark Suite                 ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # ─────────────────────────────────────────────────────────────────────────────
    # CONFIGURATION  ← only section you need to edit
    # ─────────────────────────────────────────────────────────────────────────────
    DATASET_PATH = "database/"  # output of build_mitdb_dataset.py
    SAMPLE_DURATION_S = 60 * 60  # seconds of signal to pull per segment
    N_SEGMENTS = "all"  # (measure, device) pairs to benchmark
    # MIT-BIH has 48 records × 2 channels
    # = 96 possible pairs; use 'all' for all
    # ─────────────────────────────────────────────────────────────────────────────

    # ── Load data ───────────────────────────────────────────────────────────
    segments = load_segments_from_atriumdb(DATASET_PATH, N_SEGMENTS, SAMPLE_DURATION_S)

    # ── Check on-disk size of native AtriumDB .tsc files ────────────────────
    ondisk_bytes = measure_atriumdb_ondisk(DATASET_PATH)
    if ondisk_bytes is not None:
        print(
            f"[INFO] Total .tsc on-disk data size: "
            f"{ondisk_bytes / 1024 / 1024:.2f} MB\n"
        )

    # ── C++ codec wrapper factories ──────────────────────────────────────────

    def _make_compress(fn, mode):
        def wrapper(values: np.ndarray):
            return fn(values.astype(np.int64), mode)

        return wrapper

    def _make_decompress(fn, mode):
        def wrapper(data: bytes, dtype, n: int):
            arr = np.frombuffer(data, dtype=np.uint8)
            return fn(arr, n, mode)

        return wrapper

    def _cpp(name, compress_fn, decompress_fn, mode, note):
        return CompressorSpec(
            name=name,
            compress_fn=_make_compress(compress_fn, mode),
            decompress_fn=_make_decompress(decompress_fn, mode),
            note=f"[C++] {note}",
        )

    preprocessors = [
        ("RAW", _codecs.Mode.RAW),
        ("DPCM", _codecs.Mode.DPCM),
        ("DOD", _codecs.Mode.DOD),
        ("XOR", _codecs.Mode.XOR),
        ("BYTE SHUFFLE", _codecs.Mode.BYTE_SHUFFLE),
    ]

    encoders = [
        ("BZip2", _codecs.bzip2_compress, _codecs.bzip2_decompress),
        ("Zstd", _codecs.zstd_compress, _codecs.zstd_decompress),
        ("Golomb", _codecs.golomb_compress, _codecs.golomb_decompress),
        ("LZMA", _codecs.lzma_compress, _codecs.lzma_decompress),
    ]

    COMPRESSORS: List(CompressorSpec) = [
        _cpp(f"{x[0]} + {y[0]}", y[1], y[2], x[1], f"{x[0]} + {y[0]}")
        for x in preprocessors
        for y in encoders
    ]

    # ── Run benchmarks ───────────────────────────────────────────────────────
    all_results: List[BenchResult] = []
    total = len(COMPRESSORS) * len(segments)
    done = 0

    for seg in segments:
        for spec in COMPRESSORS:
            done += 1
            label = seg.get("label", seg["measure_tag"])
            print(
                f"\r  Benchmarking [{done}/{total}] "
                f"{spec.name:<32} on '{label}' ...",
                end="",
                flush=True,
            )
            result = benchmark_one(spec, seg)
            all_results.append(result)

    print("\r" + " " * 90 + "\r")  # clear progress line

    # ── Print per-segment results ────────────────────────────────────────────
    # print_results(all_results)

    # ── Print aggregate summary ──────────────────────────────────────────────
    print_summary(all_results)

    # ── Save JSON results ────────────────────────────────────────────────────
    out_path = Path(__file__).parent / "compression_results.json"
    with open(out_path, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2)
    print(f"[INFO] Full results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
