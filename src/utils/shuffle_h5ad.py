"""
One-time cell-shuffle of a large .h5ad file (CSR X).

Problem
-------
Training with backed=True + shuffle=True causes random HDF5 reads,
which are very slow (~0.1 it/s). A pre-shuffled file lets the DataLoader use
shuffle=False (sequential reads), which are fast while still giving effectively
random batches (cells are already in a random order on disk).

Algorithm: two-pass bucket sort  (all disk I/O is sequential)
-------------------------------------------------------------
Pass 1  Read input X sequentially in chunks of `chunk_size` rows.
        For each row, compute its output position from the permutation and write
        it to a small temp "bucket" file (one bucket = one output chunk).
        --> one large sequential read of the input.

Pass 2  For each bucket, load all its rows into memory, sort by output position,
        write contiguously to the final output file.
        --> one large sequential write of the output.

Total I/O ~= 3x file_size (read input once, write temp buckets, write output).
All reads/writes are sequential.

--------------------------------------------------

Attention!!!
pd.factorize is called on the donor ID column in order of first occurrence.
Shuffling changes that order, so donor integer IDs will differ from the original
file. This is fine for training from scratch but will break checkpoint loading
from a model trained on the unshuffled file.

--------------------------------------------------

Usage
-----
python src/utils/shuffle_h5ad.py \\
    /path/to/input.h5ad \\
    /path/to/output_shuffled.h5ad \\
    [--chunk-size 500000] \\
    [--seed 32] \\
    [--tmp-dir /path/to/tmp] \\
    [--keep-tmp]
"""

import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml", "setup.py"],
    pythonpath=True,
    dotenv=True,
)

import argparse
import shutil
import time
from pathlib import Path
from typing import Optional

import anndata
import h5py
import numpy as np
import scanpy as sc
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Bucket file I/O  (binary format, supports append)
# ---------------------------------------------------------------------------
# Each bucket file stores one or more blocks written during Pass 1.
# Block layout (all little-endian):
#   k              : int32          — number of rows in this block
#   positions      : int32 × k      — within-bucket output position for each row
#   row_nnz        : int32 × k      — NNZ count for each row
#   data           : float32 × Σnnz — concatenated non-zero values
#   col_indices    : int32  × Σnnz  — concatenated column indices

def _write_bucket_block(
    fh,
    positions: np.ndarray,
    row_nnz: np.ndarray,
    data: np.ndarray,
    col_indices: np.ndarray,
) -> None:
    k = len(positions)
    np.array([k], dtype=np.int32).tofile(fh)
    positions.astype(np.int32, copy=False).tofile(fh)
    row_nnz.astype(np.int32, copy=False).tofile(fh)
    data.astype(np.float32, copy=False).tofile(fh)
    col_indices.astype(np.int32, copy=False).tofile(fh)


def _read_bucket_file(bucket_path: Path):
    """Return (positions, row_nnz, data, col_indices) as concatenated numpy arrays."""
    all_pos, all_nnz, all_data, all_idx = [], [], [], []
    with open(bucket_path, "rb") as fh:
        while True:
            raw = fh.read(4)
            if not raw:
                break
            k = int(np.frombuffer(raw, dtype=np.int32)[0])
            positions  = np.frombuffer(fh.read(k * 4), dtype=np.int32).copy()
            row_nnz    = np.frombuffer(fh.read(k * 4), dtype=np.int32).copy()
            total_nnz  = int(row_nnz.sum())
            data       = np.frombuffer(fh.read(total_nnz * 4), dtype=np.float32).copy()
            col_idx    = np.frombuffer(fh.read(total_nnz * 4), dtype=np.int32).copy()
            all_pos.append(positions)
            all_nnz.append(row_nnz)
            all_data.append(data)
            all_idx.append(col_idx)

    if not all_pos:
        return (
            np.empty(0, np.int32), np.empty(0, np.int32),
            np.empty(0, np.float32), np.empty(0, np.int32),
        )
    return (
        np.concatenate(all_pos),
        np.concatenate(all_nnz),
        np.concatenate(all_data),
        np.concatenate(all_idx),
    )


# ---------------------------------------------------------------------------
# Pass 1: sequential read --> fan-out to bucket files
# ---------------------------------------------------------------------------

def _pass1(
    input_path: str,
    tmp_dir: Path,
    inv_perm: np.ndarray,
    chunk_size: int,
    n_obs: int,
    n_buckets: int,
) -> None:
    print("Pass 1: reading input sequentially, distributing rows to bucket files ...")
    n_digits = len(str(n_obs))

    # Keep all bucket file handles open throughout Pass 1 to avoid
    # repeated open/close overhead
    bucket_fhs = [
        open(tmp_dir / f"bucket_{b:05d}.bin", "wb") for b in range(n_buckets)
    ]
    try:
        with h5py.File(input_path, "r") as f:
            indptr = f["X/indptr"][:]  # (n_obs+1,) — small, read once

            for in_start in range(0, n_obs, chunk_size):
                t0 = time.time()
                in_end = min(in_start + chunk_size, n_obs)
                chunk_len = in_end - in_start

                # One contiguous read per chunk (fast sequential I/O)
                d_start = int(indptr[in_start])
                d_end   = int(indptr[in_end])
                data_block = f["X/data"][d_start:d_end]
                idx_block  = f["X/indices"][d_start:d_end]

                # Map each row in this chunk to its output bucket and position
                global_rows  = np.arange(in_start, in_end, dtype=np.int64)
                out_positions = inv_perm[global_rows]                    # global output pos
                bucket_ids    = (out_positions // chunk_size).astype(np.int32)
                within_pos    = (out_positions %  chunk_size).astype(np.int32)

                # Row boundaries inside data_block
                row_starts = (indptr[global_rows]     - d_start).astype(np.int64)
                row_ends   = (indptr[global_rows + 1] - d_start).astype(np.int64)
                row_nnz_all = (row_ends - row_starts).astype(np.int32)

                # Fan out: one write per non-empty bucket
                for b in np.unique(bucket_ids):
                    mask      = bucket_ids == b
                    sel_starts = row_starts[mask]
                    sel_nnz    = row_nnz_all[mask]

                    sel_data = np.concatenate(
                        [data_block[rs: rs + nz] for rs, nz in zip(sel_starts, sel_nnz)]
                    ) if sel_nnz.sum() > 0 else np.empty(0, np.float32)

                    sel_idx = np.concatenate(
                        [idx_block[rs: rs + nz] for rs, nz in zip(sel_starts, sel_nnz)]
                    ) if sel_nnz.sum() > 0 else np.empty(0, np.int32)

                    _write_bucket_block(
                        bucket_fhs[b], within_pos[mask], sel_nnz, sel_data, sel_idx
                    )

                print(
                    f"  [{in_end:{n_digits}d}/{n_obs}]  "
                    f"{in_end / n_obs * 100:5.1f}%  "
                    f"{time.time() - t0:.1f}s",
                    flush=True,
                )
    finally:
        for fh in bucket_fhs:
            fh.close()

    print("Pass 1 complete.\n")


# ---------------------------------------------------------------------------
# Pass 2: merge-sort buckets --> sequential write to output
# ---------------------------------------------------------------------------

def _pass2(
    input_path: str,
    output_path: str,
    tmp_dir: Path,
    perm: np.ndarray,
    n_buckets: int,
    chunk_size: int,
    n_obs: int,
) -> None:
    print("Pass 2: sorting buckets and writing output sequentially ...")
    n_digits_b = len(str(n_buckets))

    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "a") as f_out:

        # Remove the empty placeholder X written during the metadata step
        if "X" in f_out:
            del f_out["X"]

        # Compute new indptr for the permuted row ordering
        indptr       = f_in["X/indptr"][:]
        row_lengths  = np.diff(indptr).astype(np.int64)
        new_row_lengths = row_lengths[perm]
        new_indptr   = np.zeros(n_obs + 1, dtype=np.int64)
        np.cumsum(new_row_lengths, out=new_indptr[1:])
        total_nnz = int(new_indptr[-1])

        data_dtype = f_in["X/data"].dtype
        idx_dtype  = f_in["X/indices"].dtype

        # Create output X group with the same encoding attributes
        x_grp = f_out.create_group("X")
        for k, v in f_in["X"].attrs.items():
            x_grp.attrs[k] = v
        x_grp.create_dataset("indptr", data=new_indptr)
        data_ds = x_grp.create_dataset(
            "data",    shape=(total_nnz,), dtype=data_dtype, chunks=(2_000_000,)
        )
        idx_ds = x_grp.create_dataset(
            "indices", shape=(total_nnz,), dtype=idx_dtype,  chunks=(2_000_000,)
        )

        nnz_written = 0
        for b in range(n_buckets):
            t0 = time.time()
            bf = tmp_dir / f"bucket_{b:05d}.bin"
            if not bf.exists():
                continue

            positions, row_nnz, data, col_indices = _read_bucket_file(bf)
            if len(positions) == 0:
                continue

            # Sort rows within bucket by output position
            order = np.argsort(positions, kind="stable")

            # Compute original (pre-sort) row boundaries in the data arrays
            orig_row_starts = np.zeros(len(row_nnz) + 1, dtype=np.int64)
            np.cumsum(row_nnz, out=orig_row_starts[1:])

            # Reorder data and col_indices to match sorted row order
            sorted_data = np.concatenate(
                [data[orig_row_starts[i]: orig_row_starts[i + 1]] for i in order]
            ) if total_nnz > 0 else np.empty(0, data_dtype)

            sorted_idx = np.concatenate(
                [col_indices[orig_row_starts[i]: orig_row_starts[i + 1]] for i in order]
            ) if total_nnz > 0 else np.empty(0, idx_dtype)

            # One contiguous write per bucket (sequential output I/O)
            chunk_nnz = len(sorted_data)
            data_ds[nnz_written: nnz_written + chunk_nnz] = sorted_data
            idx_ds [nnz_written: nnz_written + chunk_nnz] = sorted_idx
            nnz_written += chunk_nnz

            print(
                f"  [bucket {b + 1:{n_digits_b}d}/{n_buckets}]  "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )

    print(f"Pass 2 complete. Total NNZ written: {nnz_written:,}\n")


# ---------------------------------------------------------------------------
# Metadata: write permuted obs + var + uns to output (placeholder X)
# ---------------------------------------------------------------------------

def _write_metadata(adata: anndata.AnnData, perm: np.ndarray, output_path: str) -> None:
    print("Writing permuted metadata (obs, var, uns) to output ...")
    obs_permuted = adata.obs.iloc[perm].copy()
    meta = anndata.AnnData(
        X=csr_matrix((len(perm), adata.n_vars), dtype=np.float32),  # placeholder, replaced in Pass 2
        obs=obs_permuted,
        var=adata.var.copy(),
        uns=dict(adata.uns) if adata.uns else {},
    )
    meta.write_h5ad(output_path)
    print("  Metadata written.\n")


def shuffle_h5ad(
    input_path: str,
    output_path: str,
    chunk_size: int = 50_000,
    seed: int = 42,
    tmp_dir: Optional[str] = None,
    keep_tmp: bool = False,
) -> None:
    t_total = time.time()
    rng = np.random.default_rng(seed)

    print(f"Input:  {input_path}")
    print(f"Output: {output_path}\n")

    # --- Read metadata ---
    print("Reading metadata (backed mode) ...")
    adata = sc.read_h5ad(input_path, backed="r")
    n_obs, n_vars = adata.n_obs, adata.n_vars
    print(f"  n_obs={n_obs:,}  n_vars={n_vars:,}\n")

    # --- Generate and save permutation ---
    perm = rng.permutation(n_obs)
    inv_perm = np.argsort(perm)  # inv_perm[i] = output row for input row i

    perm_path = Path(output_path).with_suffix(".perm.npy")
    np.save(perm_path, perm)
    print(f"Permutation saved → {perm_path}\n")

    # --- Set up temp directory ---
    tmp_root = Path(tmp_dir) if tmp_dir else Path(output_path).parent / ".shuffle_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    n_buckets = (n_obs + chunk_size - 1) // chunk_size
    print(f"Buckets: {n_buckets}  (chunk_size={chunk_size:,})")
    print(f"Tmp dir: {tmp_root}\n")

    # --- Step 1: write permuted obs/var/uns with placeholder X ---
    _write_metadata(adata, perm, output_path)

    # --- Step 2: two-pass shuffle of X ---
    _pass1(input_path, tmp_root, inv_perm, chunk_size, n_obs, n_buckets)
    _pass2(input_path, output_path, tmp_root, perm, n_buckets, chunk_size, n_obs)

    # --- Clean up temp files ---
    if not keep_tmp:
        shutil.rmtree(tmp_root)
        print("Temp files removed.")
    else:
        print(f"Temp files kept at {tmp_root}")

    elapsed = time.time() - t_total
    print(f"\nFinished in {elapsed / 60:.1f} min")
    print(f"\nNext steps:")
    print(f"  1. Update `adata` in configs/datamodule/Cardinal_cis-corrected.yaml")
    print(f"     to: {output_path}")
    print(f"  2. Set `shuffle: False` in the same config.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input",  help="Path to input .h5ad file.")
    parser.add_argument("output", help="Path for shuffled output .h5ad file.")
    parser.add_argument(
        "--chunk-size", type=int, default=50_000,
        help="Rows per bucket (default: 50000). Larger = more RAM per Pass 2 bucket.",
    )
    parser.add_argument("--seed",     type=int, default=42)
    parser.add_argument(
        "--tmp-dir", default=None,
        help="Directory for temp bucket files (default: <output_parent>/.shuffle_tmp).",
    )
    parser.add_argument(
        "--keep-tmp", action="store_true",
        help="Keep temp bucket files after completion (useful for debugging).",
    )
    args = parser.parse_args()

    shuffle_h5ad(
        input_path=args.input,
        output_path=args.output,
        chunk_size=args.chunk_size,
        seed=args.seed,
        tmp_dir=args.tmp_dir,
        keep_tmp=args.keep_tmp,
    )


if __name__ == "__main__":
    main()
