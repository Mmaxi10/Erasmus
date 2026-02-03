"""Clean FFIEC panel CSVs produced by DataFormatting.py.

Run from the repository root (or any directory) with:
    python3 DataCleaning.py

By default this reads CSVs under DATA/FFIEC (csv) and writes a single combined
dataset under DATA. Toggle TRIAL_MODE below to run a single year.
"""

import argparse
import re
from pathlib import Path

try:
    import pandas as pd
except ImportError as exc:  # pragma: no cover - makes the dependency explicit for users
    raise SystemExit(
        "pandas is required to run this script. Install dependencies from requirements.txt."
    ) from exc


# ---- Editable settings (adjust these before running) ----

# Default paths assume this script lives in .../Banking Seminar/Data Processing/.
BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = BASE_DIR / "DATA" / "FFIEC (csv)"
DEFAULT_OUTPUT_DIR = BASE_DIR / "DATA"

# Tokens treated as missing values (case-insensitive).
MISSING_TOKENS = {"", "nan", "na", "null"}

# Drop any column with missingness >= this fraction.
COLUMN_MISSINGNESS_THRESHOLD = 0.50

# Treat numeric zeros as empty when removing fully empty/zero rows.
TREAT_ZERO_AS_EMPTY = True

# Remove rows only if every column is empty/zero under the rule above.
DROP_ROWS_ALL_EMPTY_OR_ZERO = True

# Remove rows if any column is missing.
# Keep False so earlier years aren't dropped due to non-core columns (e.g., interest expense on deposits).
DROP_ROWS_WITH_ANY_MISSING = False

# Merge duplicate columns that share the same base name (e.g., with codes).
MERGE_DUPLICATE_COLUMNS = True

# Pick the most complete duplicate column as the canonical source.
USE_MOST_COMPLETE_CANONICAL = True

# Reorder columns to match KEEP_COLUMNS after merging.
ORDER_COLUMNS = True

# Print merge-fill counts after duplicate-column merge.
REPORT_MERGE_STATS = True
REPORT_MERGE_DETAILS = False

# Trial mode runs a single year (matched by substring in filename stem).
TRIAL_MODE = False
TRIAL_YEAR = "2011"  # Example: "2011". Set TRIAL_MODE = False to run all years.

# Keep only these columns; set to [] to disable filtering.
KEEP_COLUMNS = [
    "Reporting Period End Date",
    "IDRSSD",
    "TOTAL ASSETS (RCON2170)",
    "TOTAL DEPOSITS (RCON2200)",
    "NET INTEREST INCOME",
    "INTEREST EXPENSE, INTEREST ON DEPOSI",
    "INTEREST EXPENSE, INTEREST ON DEPOSI (RIADHK04)",
    "NON-INTEREST BEARING DEPOSITS",
    "NON-INTEREST BEARING DEPOSITS (RCON6631)",
    "NONINTEREST-BEARING BALS&CURR&COIN",
    "NONINTEREST-BEARING BALS&CURR&COIN (RCON0081)",
    "TOTAL DEPOSITS",
    "TOTAL ASSETS",
]

# Columns that should stay as identifiers (strings).
ID_COLUMNS = [
    "IDRSSD",
]

# Columns to parse as dates (date-only).
DATE_ONLY_COLUMNS = [
    "Reporting Period End Date",
]

# Columns to parse as full datetimes.
DATETIME_COLUMNS = []

# Write a combined dataset across all processed years.
WRITE_COMBINED_DATASET = True
COMBINED_FILENAME = "ffiec_combined.csv"

# Skip writing per-year cleaned datasets (combined only).
WRITE_YEARLY_DATASETS = False


MISSING_TOKENS_LOWER = {token.lower() for token in MISSING_TOKENS}
KEEP_COLUMNS_SET = set(KEEP_COLUMNS)
ID_COLUMNS_SET = set(ID_COLUMNS)
DATE_ONLY_COLUMNS_SET = set(DATE_ONLY_COLUMNS)
DATETIME_COLUMNS_SET = set(DATETIME_COLUMNS)


def _empty_or_zero_mask(series: pd.Series) -> pd.Series:
    """Return True for cells treated as empty: missing tokens and optional zeros."""
    missing = series.isna()
    text = series.astype(str).str.strip()
    missing = missing | text.str.lower().isin(MISSING_TOKENS_LOWER)
    if not TREAT_ZERO_AS_EMPTY:
        return missing
    numeric = pd.to_numeric(text, errors="coerce")
    return missing | numeric.eq(0)


def _missing_mask(series: pd.Series) -> pd.Series:
    """Return True for cells treated as missing (no numeric-zero logic here)."""
    missing = series.isna()
    text = series.astype(str).str.strip()
    return missing | text.str.lower().isin(MISSING_TOKENS_LOWER)


def _base_name(name: str) -> str:
    """Return the base name with trailing parenthetical codes removed."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()


def merge_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Coalesce columns that share the same base name and count fill direction."""
    stats: dict[str, dict[str, object]] = {}
    groups: dict[str, list[str]] = {}
    for col in df.columns:
        groups.setdefault(_base_name(col), []).append(col)
    for base, cols in groups.items():
        if len(cols) < 2:
            continue
        missing = {col: _missing_mask(df[col]) for col in cols}
        missing_counts = {col: int(mask.sum()) for col, mask in missing.items()}
        if USE_MOST_COMPLETE_CANONICAL:
            canonical_source = min(
                cols, key=lambda col: (missing_counts[col], col != base)
            )
        else:
            canonical_source = base if base in cols else cols[0]
        canonical_name = base if base in KEEP_COLUMNS_SET else canonical_source
        base_missing = missing[canonical_source]
        other_present = pd.concat(
            [~missing[col] for col in cols if col != canonical_source], axis=1
        ).any(axis=1)
        other_missing = pd.concat(
            [missing[col] for col in cols if col != canonical_source], axis=1
        ).any(axis=1)
        stats[canonical_name] = {
            "rows": int(len(df)),
            "canonical_source": canonical_source,
            "missing_counts": missing_counts,
            "canonical_fill": int((base_missing & other_present).sum()),
            "other_fill": int((other_missing & ~base_missing).sum()),
        }
        merged = None
        for col in cols:
            series = df[col].mask(_missing_mask(df[col]))
            merged = series if merged is None else merged.combine_first(series)
        df[canonical_name] = merged
        df = df.drop(columns=[col for col in cols if col != canonical_name])
    df.attrs["merge_stats"] = stats
    return df


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Order columns to match KEEP_COLUMNS, collapsing duplicates by base name."""
    ordered: list[str] = []
    seen: set[str] = set()
    for col in KEEP_COLUMNS:
        base = _base_name(col)
        if base in df.columns and base not in seen:
            ordered.append(base)
            seen.add(base)
        elif col in df.columns and col not in seen:
            ordered.append(col)
            seen.add(col)
    ordered.extend(col for col in df.columns if col not in seen)
    return df.loc[:, ordered]


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Parse dates, keep IDs as strings, and coerce the rest to numeric."""
    for col in df.columns:
        if col in ID_COLUMNS_SET:
            df[col] = df[col].astype("string").str.strip()
        elif col in DATE_ONLY_COLUMNS_SET:
            df[col] = pd.to_datetime(df[col], errors="coerce").dt.normalize()
        elif col in DATETIME_COLUMNS_SET:
            df[col] = pd.to_datetime(df[col], errors="coerce")
        else:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Filter columns, merge duplicates, coerce types, and drop missing rows."""
    if KEEP_COLUMNS_SET:
        df = df.loc[:, [col for col in df.columns if col in KEEP_COLUMNS_SET]]
    if MERGE_DUPLICATE_COLUMNS:
        df = merge_duplicate_columns(df)
    else:
        df.attrs["merge_stats"] = {}
    if ORDER_COLUMNS:
        df = order_columns(df)
    df = coerce_types(df)
    missing = df.apply(_missing_mask)
    protected = {_base_name(col) for col in KEEP_COLUMNS} if KEEP_COLUMNS_SET else set()
    drop_cols = [
        col
        for col, frac in missing.mean(axis=0).items()
        if frac >= COLUMN_MISSINGNESS_THRESHOLD and col not in protected
    ]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    if df.empty:
        return df
    missing = df.apply(_missing_mask)
    if DROP_ROWS_WITH_ANY_MISSING:
        df = df.loc[~missing.any(axis=1)].copy()
    if df.empty or not DROP_ROWS_ALL_EMPTY_OR_ZERO:
        return df
    row_mask = df.apply(_empty_or_zero_mask).all(axis=1)
    return df.loc[~row_mask].copy()


def select_csv_files(input_dir: Path) -> list[Path]:
    """Return either all CSVs or only the single trial-year CSV."""
    files = sorted(p for p in input_dir.glob("*.csv") if p.is_file())
    if not TRIAL_MODE:
        return files
    if not TRIAL_YEAR:
        raise ValueError("TRIAL_YEAR must be set when TRIAL_MODE is True.")
    trial_files = [p for p in files if TRIAL_YEAR in p.stem]
    if not trial_files:
        raise FileNotFoundError(
            f"No CSV files matched trial year {TRIAL_YEAR} in {input_dir}"
        )
    return trial_files


def clean_all_files(input_dir: Path, output_dir: Path) -> None:
    """Clean every CSV in input_dir and write the results to output_dir."""
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_files = select_csv_files(input_dir)
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    combined_frames: list[pd.DataFrame] = []
    for path in csv_files:
        # Read as strings to keep IDs intact and preserve raw missingness.
        df = pd.read_csv(path, dtype=str, encoding="latin1", low_memory=False)
        if KEEP_COLUMNS_SET:
            present_bases = {_base_name(col) for col in df.columns}
            missing_keep = [
                col
                for col in KEEP_COLUMNS
                if col not in df.columns and _base_name(col) not in present_bases
            ]
            if missing_keep:
                print(f"{path.name}: missing keep columns -> {missing_keep}")
        original_rows, original_cols = df.shape

        cleaned = clean_dataframe(df)
        cleaned_rows, cleaned_cols = cleaned.shape
        if REPORT_MERGE_STATS:
            stats = cleaned.attrs.get("merge_stats", {})
            total_fills = (
                sum(entry["canonical_fill"] for entry in stats.values())
                if stats
                else 0
            )
            print(f"{path.name}: canonical fills -> {total_fills}")
            if REPORT_MERGE_DETAILS and stats:
                for col, entry in stats.items():
                    print(
                        f"{path.name}: merge {col} rows={entry['rows']} "
                        f"source={entry['canonical_source']} "
                        f"canonical_fill={entry['canonical_fill']} "
                        f"other_fill={entry['other_fill']} "
                        f"missing={entry['missing_counts']}"
                    )

        if WRITE_YEARLY_DATASETS:
            output_path = output_dir / path.name
            cleaned.to_csv(output_path, index=False)
        print(
            f"{path.name}: {original_rows}x{original_cols} -> "
            f"{cleaned_rows}x{cleaned_cols}"
            + (" (combined only)" if not WRITE_YEARLY_DATASETS else "")
        )
        if WRITE_COMBINED_DATASET:
            combined_frames.append(cleaned)

    if WRITE_COMBINED_DATASET and combined_frames:
        combined = pd.concat(combined_frames, ignore_index=True, sort=False)
        combined_path = output_dir / COMBINED_FILENAME
        combined.to_csv(combined_path, index=False)
        print(f"Wrote combined dataset -> {combined_path} ({len(combined)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove fully empty/zero rows and columns from FFIEC CSVs."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing yearly CSVs (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Where to write cleaned CSVs (default: {DEFAULT_OUTPUT_DIR})",
    )
    args = parser.parse_args()

    clean_all_files(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
