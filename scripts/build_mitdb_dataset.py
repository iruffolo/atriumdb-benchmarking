"""
build_mitdb_dataset.py
======================
Downloads the MIT-BIH Arrhythmia Database from PhysioNet and ingests every
record into a local AtriumDB dataset ready for compression benchmarking.

MIT-BIH database facts
----------------------
  - 48 records (100–234, non-contiguous numbering)
  - 2 ECG channels each: typically MLII + V1/V2/V4/V5
  - Sampling rate: 360 Hz
  - Duration: ~30 minutes per record (~650,000 samples per channel)
  - Digital resolution: 11-bit (stored as int64 in AtriumDB)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import wfdb
from atriumdb import AtriumSDK
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def fmt_size(n_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} GB"


def _get_or_create_measure(sdk, tag: str, freq_nhz: int, units: str) -> int:
    """Return existing measure_id or create a new measure."""
    mid = sdk.get_measure_id(
        measure_tag=tag,
        freq=freq_nhz,
        units=units,
        freq_units="nHz",
    )
    if mid is None:
        mid = sdk.insert_measure(
            measure_tag=tag,
            freq=freq_nhz,
            units=units,
            freq_units="nHz",
        )
    return mid


def _get_or_create_device(sdk, tag: str) -> int:
    """Return existing device_id or create a new device."""
    did = sdk.get_device_id(device_tag=tag)
    if did is None:
        did = sdk.insert_device(device_tag=tag)
    return did


# ──────────────────────────────────────────────────────────────────────────────
# INGEST ONE RECORD
# ──────────────────────────────────────────────────────────────────────────────
def ingest_record(sdk, record_name: str, quiet=False) -> dict:
    """
    Read one MIT-BIH record and write all signal channels into AtriumDB.

    AtriumDB write_segment signature (verified against SDK source):
        write_segment(measure_id, device_id, segment_values, start_time,
                      freq=None, freq_units=None, time_units=None, ...)

    start_time is interpreted in `time_units` (we use 'ns' = nanoseconds).
    freq is in `freq_units` (we use 'nHz' = nanohertz).

    Returns a stats dict describing what was written.
    """

    # load digital signal from MIT-BIH
    record = wfdb.rdrecord(record_name, pn_dir="mitdb", return_res=64, physical=False)

    # d_signal: (n_samples, n_channels), dtype int64
    d_signal = record.d_signal
    n_samples = record.sig_len
    n_channels = record.n_sig
    fs_hz = float(record.fs)  # 360.0 Hz for all MIT-BIH records
    freq_nhz = int(fs_hz * 1_000_000_000)  # nanohertz

    device_id = _get_or_create_device(sdk, record_name)

    stats = {
        "record": record_name,
        "fs_hz": fs_hz,
        "freq_nhz": freq_nhz,
        "n_samples": n_samples,
        "channels": [],
        "total_samples": 0,
    }

    for ch_idx in range(n_channels):
        sig_name = record.sig_name[ch_idx]
        units = (record.units or ["mV"] * n_channels)[ch_idx]
        values = np.ascontiguousarray(d_signal[:, ch_idx], dtype=np.int64)

        measure_id = _get_or_create_measure(sdk, sig_name, freq_nhz, units)

        # write_segment anchor: t=0 ns, freq in nHz
        sdk.write_segment(
            measure_id=measure_id,
            device_id=device_id,
            segment_values=values,
            start_time=0,  # nanoseconds (time_units='ns')
            freq=freq_nhz,
            freq_units="nHz",
            time_units="ns",
        )

        stats["channels"].append(
            {
                "sig_name": sig_name,
                "units": units,
                "measure_id": measure_id,
                "device_id": device_id,
                "n_samples": n_samples,
            }
        )
        stats["total_samples"] += n_samples

        if not quiet:
            print(
                f"      ├─ {sig_name:<6}  {n_samples:>9,} samples  "
                f"units={units:<4}  measure_id={measure_id}  device_id={device_id}"
            )

    return stats


# ──────────────────────────────────────────────────────────────────────────────
# VERIFY ONE RECORD
# ──────────────────────────────────────────────────────────────────────────────
def verify_record(sdk, stats: dict) -> bool:
    """
    Read back each channel from AtriumDB and assert sample count matches.
    Returns True if all channels pass.
    """
    fs_hz = stats["fs_hz"]
    ok = True

    for ch in stats["channels"]:
        n_expected = ch["n_samples"]
        # end time = last sample's timestamp + 1 period (open interval)
        period_ns = int(1e18 // stats["freq_nhz"])
        end_ns = n_expected * period_ns

        _, times, values = sdk.get_data(
            measure_id=ch["measure_id"],
            device_id=ch["device_id"],
            start_time_n=0,
            end_time_n=end_ns + period_ns,
        )
        n_got = len(values) if values is not None else 0

        if n_got != n_expected:
            print(
                f"    [WARN] {stats['record']}/{ch['sig_name']}: "
                f"expected {n_expected} samples, read back {n_got}"
            )
            ok = False

    return ok


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
def print_summary(
    dataset_path: Path,
    all_stats: list,
    failed: list,
    elapsed: float,
    total_samples: int,
):
    print(f"\n{'═'*64}")
    print("  MIT-BIH → AtriumDB Ingest Summary")
    print(f"{'═'*64}")

    n_ok = len(all_stats)
    n_total = n_ok + len(failed)
    print(f"  Records ingested  : {n_ok} / {n_total}")
    print(f"  Total samples     : {total_samples:,}")

    # Count unique measures and devices
    all_mids = set()
    all_dids = set()
    for s in all_stats:
        all_dids.add(s["record"])
        for ch in s["channels"]:
            all_mids.add((ch["sig_name"], ch["measure_id"]))
    print(f"  Unique measures   : {len(all_mids)}")
    print(f"  Unique devices    : {len(all_dids)}")
    print(
        f"  Elapsed time      : {elapsed:.1f} s  "
        f"({total_samples / max(elapsed, 0.001) / 1e6:.1f} M samples/s)"
    )

    # On-disk storage stats
    tsc_files = list(dataset_path.rglob("*.tsc"))
    if tsc_files:
        tsc_bytes = sum(f.stat().st_size for f in tsc_files)
        raw_bytes = total_samples * 8  # int64 → 8 bytes/sample
        ratio = raw_bytes / tsc_bytes if tsc_bytes else 0.0
        print(f"\n  .tsc data files   : {len(tsc_files)}")
        print(f"  On-disk (tsc)     : {fmt_size(tsc_bytes)}")
        print(f"  Uncompressed est. : {fmt_size(raw_bytes)}")
        print(f"  AtriumDB ratio    : {ratio:.2f}x")

    if failed:
        print(f"\n  Failed records ({len(failed)}):")
        for name, err in failed:
            print(f"    {name}: {err}")

    print(f"\n  Dataset path : {dataset_path.resolve()}")
    print()


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║         MIT-BIH Arrhythmia Database → AtriumDB Builder       ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    dataset_path = Path("../database/")

    # create or reopen dataset
    if any(dataset_path.iterdir()):
        print(f"\n[INFO] Reopening existing dataset: {dataset_path}")
        sdk = AtriumSDK(dataset_location=str(dataset_path))
    else:
        print(f"\n[INFO] Creating new dataset: {dataset_path}")
        sdk = AtriumSDK.create_dataset(dataset_location=str(dataset_path))

    # get record list
    print("[INFO] Fetching record list from PhysioNet (mitdb)...")
    try:
        record_names = wfdb.get_record_list("mitdb")
    except Exception as e:
        print(f"[ERROR] Could not fetch record list: {e}")
        sys.exit(1)
    print(f"[INFO] {len(record_names)} records found.\n")

    # ingest loop
    t_start = time.perf_counter()
    all_stats = []
    failed = []
    total_samples = 0

    bar = tqdm(record_names, desc="Records", unit="rec", ncols=72)

    for record_name in bar:
        bar.set_postfix({"current": record_name})

        tqdm.write(f"\n  ► {record_name}")

        stats = ingest_record(
            sdk,
            record_name,
            quiet=False,
        )
        ok = verify_record(sdk, stats)
        status = "✓ verified" if ok else "✗ mismatch"
        tqdm.write(f"      └─ {status}")

        all_stats.append(stats)
        total_samples += stats["total_samples"]

    bar.close()
    elapsed = time.perf_counter() - t_start

    print_summary(dataset_path, all_stats, failed, elapsed, total_samples)


if __name__ == "__main__":
    main()
