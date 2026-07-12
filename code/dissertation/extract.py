from __future__ import annotations

import argparse
import re
from pathlib import Path

from extract_core import (
    ColumnSpec,
    ExtractionPlan,
    TableSpec,
    extract_pdf_tables,
    join_lines,
    normalize_spaces,
    remove_whitespace,
    write_csv_tables,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF = PROJECT_ROOT / "data" / "mmc1.pdf"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "extracted" / "clean"


def mmc1_extraction_plan() -> ExtractionPlan:
    return ExtractionPlan(
        table_caption_pattern=re.compile(r"Table S(\d+)\."),
        tables={
            1: TableSpec(
                table_id=1,
                columns=[
                    ColumnSpec("No."),
                    ColumnSpec("Sequence", remove_whitespace),
                    ColumnSpec("Nucleotide Availability"),
                    ColumnSpec("Nucleotide Strong (1) or Weak (0)"),
                    ColumnSpec("k1"),
                    ColumnSpec("k2"),
                ],
                skip_joined_header_fragments=("Training Features", "Target Features"),
            ),
            2: TableSpec(
                table_id=2,
                columns=[
                    ColumnSpec("Gene", join_lines),
                    ColumnSpec("Sequence", remove_whitespace),
                ],
            ),
            3: TableSpec(
                table_id=3,
                columns=[
                    ColumnSpec("No."),
                    ColumnSpec("Sequence", remove_whitespace),
                    ColumnSpec("Nucleotide Availability"),
                ],
            ),
            4: TableSpec(
                table_id=4,
                columns=[
                    ColumnSpec("No."),
                    ColumnSpec("Sequence", remove_whitespace),
                    ColumnSpec("Predicted k1", normalize_spaces),
                ],
            ),
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract dissertation source tables from mmc1.pdf into CSV files."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help=f"PDF to extract tables from. Defaults to {DEFAULT_PDF}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for the CSV files. Defaults to {DEFAULT_OUTPUT_DIR}",
    )
    return parser.parse_args()


def extract_mmc1_tables(pdf_path: Path, output_dir: Path) -> list[Path]:
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    extracted_tables = extract_pdf_tables(pdf_path, mmc1_extraction_plan())
    return write_csv_tables(extracted_tables, output_dir)


def main() -> None:
    args = parse_args()
    written_paths = extract_mmc1_tables(args.pdf.resolve(), args.output_dir.resolve())
    for path in written_paths:
        row_count = sum(1 for _ in path.open(encoding="utf-8")) - 1
        print(f"Wrote {path} ({row_count} rows)")


if __name__ == "__main__":
    main()
