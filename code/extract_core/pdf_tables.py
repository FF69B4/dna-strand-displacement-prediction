from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import pdfplumber


CellTransform = Callable[[str | None], str]


def normalize_spaces(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value).strip()


def remove_whitespace(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", "", value.strip())


def join_lines(value: str | None, separator: str = "; ") -> str:
    if value is None:
        return ""
    return separator.join(line.strip() for line in value.splitlines() if line.strip())


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    transform: CellTransform = normalize_spaces


@dataclass(frozen=True)
class TableSpec:
    table_id: int
    columns: list[ColumnSpec]
    output_name: str | None = None
    skip_joined_header_fragments: tuple[str, ...] = ()

    @property
    def header(self) -> list[str]:
        return [column.name for column in self.columns]

    @property
    def csv_name(self) -> str:
        return self.output_name or f"table_{self.table_id}.csv"


@dataclass(frozen=True)
class ExtractionPlan:
    tables: dict[int, TableSpec]
    table_caption_pattern: re.Pattern[str] = field(
        default_factory=lambda: re.compile(r"Table\s+(\d+)\.")
    )

    def detect_table_id(self, page_text: str) -> int | None:
        match = self.table_caption_pattern.search(page_text)
        if match:
            return int(match.group(1))
        return None


@dataclass
class ExtractedTable:
    spec: TableSpec
    rows: list[list[str]] = field(default_factory=list)


def is_header_row(spec: TableSpec, row: list[str]) -> bool:
    normalized = [normalize_spaces(cell) for cell in row]
    expected = spec.header

    if normalized == expected:
        return True
    if len(normalized) == len(expected) and not normalized[0] and normalized[1:] == expected[1:]:
        return True

    joined = " ".join(normalized)
    return any(fragment in joined for fragment in spec.skip_joined_header_fragments)


def normalize_row(spec: TableSpec, row: list[str | None]) -> list[str]:
    aligned = list(row[: len(spec.columns)])
    if len(aligned) < len(spec.columns):
        aligned.extend([None] * (len(spec.columns) - len(aligned)))

    return [
        column.transform(value)
        for column, value in zip(spec.columns, aligned, strict=True)
    ]


def extract_pdf_tables(pdf_path: Path, plan: ExtractionPlan) -> dict[int, ExtractedTable]:
    extracted = {
        table_id: ExtractedTable(spec=spec)
        for table_id, spec in plan.tables.items()
    }
    current_table_id: int | None = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            detected_table_id = plan.detect_table_id(page_text)
            if detected_table_id is not None:
                current_table_id = detected_table_id

            if current_table_id is None or current_table_id not in plan.tables:
                continue

            spec = plan.tables[current_table_id]
            for raw_table in page.extract_tables() or []:
                for raw_row in raw_table:
                    cleaned_row = normalize_row(spec, raw_row)
                    if not any(cleaned_row):
                        continue
                    if is_header_row(spec, cleaned_row):
                        continue
                    extracted[current_table_id].rows.append(cleaned_row)

    return extracted


def write_csv_tables(extracted_tables: dict[int, ExtractedTable], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written_paths: list[Path] = []

    for table_id in sorted(extracted_tables):
        extracted = extracted_tables[table_id]
        output_path = output_dir / extracted.spec.csv_name
        with output_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(extracted.spec.header)
            writer.writerows(extracted.rows)
        written_paths.append(output_path)

    return written_paths
