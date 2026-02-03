"""Build yearly FFIEC panel CSVs from the tab-delimited raw exports.

Run from the repository root (or any directory) with:
    python3 DataFormatting.py

By default this reads the raw files under DATA/FFIEC (raw) and writes one
comma-delimited CSV per year under DATA/FFIEC (csv).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Tuple

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - makes the dependency explicit for users
    raise SystemExit(
        "pandas is required to run this script. Install dependencies from requirements.txt."
    ) from exc


# The first 13 columns are stable identifiers present in every raw file.
ID_COLUMN_COUNT = 13

# These defaults assume the current file lives in .../Banking Seminar/Data Processing/.
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = BASE_DIR / "DATA" / "FFIEC (raw)"
DEFAULT_OUTPUT_DIR = BASE_DIR / "DATA" / "FFIEC (csv)"


def _strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
    """Trim surrounding whitespace from string columns to avoid merge mismatches."""
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].str.strip()
    return df


def _read_header_lines(path: Path) -> Tuple[List[str], List[str]]:
    with path.open("r", encoding="latin1") as handle:
        codes_line = handle.readline()
        descriptions_line = handle.readline()
    if not codes_line:
        raise ValueError(f"Missing header line in {path}")
    codes = codes_line.rstrip("\n").split("\t")
    descriptions = descriptions_line.rstrip("\n").split("\t") if descriptions_line else []
    if len(descriptions) < len(codes):
        descriptions.extend([""] * (len(codes) - len(descriptions)))
    return codes, descriptions


def _looks_like_data_row(fields: List[str]) -> bool:
    if not fields:
        return False
    first = fields[0].strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", first):
        return True
    return False


def _dedupe_headers(headers: List[str], codes: List[str]) -> List[str]:
    seen: dict[str, int] = {}
    deduped: List[str] = []
    for idx, name in enumerate(headers):
        clean_name = name.strip()
        if clean_name == "":
            deduped.append(clean_name)
            continue
        if clean_name in seen:
            seen[clean_name] += 1
            code = codes[idx].strip()
            suffix = code if code and code != clean_name else str(seen[clean_name])
            deduped.append(f"{clean_name} ({suffix})")
        else:
            seen[clean_name] = 1
            deduped.append(clean_name)
    return deduped


def _build_headers(codes: List[str], descriptions: List[str]) -> Tuple[List[str], int]:
    use_descriptions = bool(descriptions) and not _looks_like_data_row(descriptions)
    if not use_descriptions:
        return [col.strip() for col in codes], 1
    headers = [
        desc.strip() if desc and desc.strip() else code.strip()
        for code, desc in zip(codes, descriptions)
    ]
    return _dedupe_headers(headers, codes), 2


def load_tab_file(path: Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load a tab-delimited FFIEC slice, clean headers, and drop empty rows."""
    codes, descriptions = _read_header_lines(path)
    headers, skip_rows = _build_headers(codes, descriptions)
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        engine="python",
        encoding="latin1",
        on_bad_lines="warn",
        header=None,
        names=headers,
        skiprows=skip_rows,
    )

    df.columns = [col.strip() for col in df.columns]
    # Remove empty placeholder columns that can appear because of trailing tabs.
    drop_columns = [
        col
        for col in df.columns
        if col.strip() == "" or (col.startswith("Unnamed") and df[col].isna().all())
    ]
    if drop_columns:
        df = df.drop(columns=drop_columns)

    df = _strip_whitespace(df)
    first_col = df.columns[0]
    first_col_series = df[first_col]
    valid_rows = (
        first_col_series.notna()
        & first_col_series.astype(str).str.strip().ne("")
        & first_col_series.astype(str).str.lower().ne("nan")
    )
    df = df.loc[valid_rows].copy()

    id_cols = list(df.columns[:ID_COLUMN_COUNT])
    return df, id_cols


def merge_year_files(files: Iterable[Path]) -> pd.DataFrame:
    """Merge all slices for a year horizontally on the shared identifier columns."""
    merged_df: pd.DataFrame | None = None
    id_cols: List[str] = []
    data_cols: List[str] = []

    for path in sorted(files):
        df, current_ids = load_tab_file(path)

        if merged_df is None:
            merged_df = df
            id_cols = current_ids
            data_cols = [c for c in merged_df.columns if c not in id_cols]
            continue

        if current_ids != id_cols:
            raise ValueError(
                f"Identifier column mismatch while merging {path.name}. "
                f"Expected {id_cols}, saw {current_ids}"
            )

        merged_df = merged_df.merge(df, on=id_cols, how="outer")
        for col in df.columns:
            if col not in id_cols and col not in data_cols:
                data_cols.append(col)

    if merged_df is None:
        raise ValueError("No data files were provided to merge.")

    ordered_columns = list(id_cols) + data_cols
    return merged_df.reindex(columns=ordered_columns)


def detect_year_from_dir(path: Path) -> str | None:
    match = re.search(r"(20\d{2})", path.name)
    return match.group(1) if match else None


def transform_all_years(raw_dir: Path, output_dir: Path) -> None:
    raw_dir = raw_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    year_dirs = [p for p in raw_dir.iterdir() if p.is_dir()]
    if not year_dirs:
        raise FileNotFoundError(f"No year directories found in {raw_dir}")

    for year_dir in sorted(year_dirs):
        year = detect_year_from_dir(year_dir)
        if not year:
            continue

        txt_files = [
            p
            for p in sorted(year_dir.glob("*.txt"))
            if p.is_file() and p.name.lower() != "readme.txt"
        ]
        if not txt_files:
            print(f"Skipping {year} (no .txt data files found).")
            continue

        if len(txt_files) > 2:
            print(
                f"Found {len(txt_files)} files for {year}; using the first 2 parts only."
            )
            txt_files = txt_files[:2]

        print(f"Processing {year} ({len(txt_files)} files)...")
        merged = merge_year_files(txt_files)

        output_path = output_dir / f"ffiec_{year}.csv"
        merged.to_csv(output_path, index=False)
        print(
            f"Wrote {output_path} with {len(merged)} rows and {len(merged.columns)} columns."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transform tab-delimited FFIEC slices into one CSV per year."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW_DIR,
        help=f"Directory containing the yearly folders (default: {DEFAULT_RAW_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write the yearly CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    transform_all_years(args.raw_dir, args.output_dir)


if __name__ == "__main__":
    main()
