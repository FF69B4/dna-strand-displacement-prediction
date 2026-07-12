from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import os
from pathlib import Path
import re
from typing import Iterable

import numpy as np

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")


VALID_BASES = {"A", "C", "G", "T", "N"}
KMER_PATTERNS = (
    re.compile(r"dna[_-]?bert[_/-]?(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)[_-]?mer(?:s)?", re.IGNORECASE),
    re.compile(r"k[-_]?(\d+)", re.IGNORECASE),
    re.compile(r"(\d+)[_-]?k\b", re.IGNORECASE),
)


@dataclass
class SequenceRecord:
    record_id: str
    sequence: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DNA-BERT embeddings from sequence data. "
            "The legacy filename is retained, but this script now creates embeddings."
        )
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="Path to a CSV/TSV, plain text, or FASTA file containing DNA sequences.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Destination .npz file. Defaults to <input>_dnabert_embeddings.npz",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        help="Optional CSV file to write the embeddings in tabular form.",
    )
    parser.add_argument(
        "--input-format",
        choices=("auto", "csv", "tsv", "text", "fasta"),
        default="auto",
        help="How to parse the input file. Defaults to auto-detection.",
    )
    parser.add_argument(
        "--model-name",
        default="zhihan1996/DNA_bert_6",
        help=(
            "Hugging Face model name or local model path. "
            "Use a DNABERT-compatible checkpoint."
        ),
    )
    parser.add_argument(
        "--model-kmer-size",
        type=int,
        help=(
            "Expected k-mer size for the model. "
            "If omitted, the script will try to infer it from the model name or local metadata."
        ),
    )
    parser.add_argument(
        "--sequence-column",
        default="Sequence",
        help="Column name containing sequences when reading CSV/TSV input.",
    )
    parser.add_argument(
        "--id-column",
        default="No.",
        help="Column name to use as the record id when reading CSV/TSV input.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for model inference.",
    )
    parser.add_argument(
        "--pooling",
        choices=("cls", "mean", "max"),
        default="mean",
        help="How to pool token embeddings into one vector per sequence.",
    )
    parser.add_argument(
        "--kmer-size",
        type=int,
        default=6,
        help=(
            "K-mer size for legacy DNABERT models. "
            "Set to 0 to feed raw sequences directly."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum tokenizer length before truncation.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Torch device to run inference on.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when loading the tokenizer and model.",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="Skip rows containing invalid sequence characters instead of failing.",
    )
    return parser.parse_args()


def detect_input_format(path: Path, requested_format: str) -> str:
    if requested_format != "auto":
        return requested_format

    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return "tsv"
    if suffix in {".fa", ".fasta", ".fna"}:
        return "fasta"
    if suffix in {".txt", ".seq"}:
        return "text"
    return "csv"


def normalize_sequence(sequence: str) -> str:
    cleaned = "".join(sequence.upper().split())
    invalid = sorted(set(cleaned) - VALID_BASES)
    if invalid:
        raise ValueError(
            f"Sequence contains unsupported bases {invalid}. "
            "Allowed bases are A, C, G, T, and N."
        )
    return cleaned


def load_delimited_records(
    path: Path,
    delimiter: str,
    sequence_column: str,
    id_column: str,
    skip_invalid: bool,
) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        if reader.fieldnames is None:
            raise ValueError(f"No header row found in {path}")
        if sequence_column not in reader.fieldnames:
            raise ValueError(
                f"Sequence column {sequence_column!r} not found in {path}. "
                f"Available columns: {reader.fieldnames}"
            )

        for row_number, row in enumerate(reader, start=2):
            raw_sequence = row.get(sequence_column, "")
            if not raw_sequence or not raw_sequence.strip():
                continue

            record_id = row.get(id_column) or str(len(records))
            try:
                sequence = normalize_sequence(raw_sequence)
            except ValueError:
                if skip_invalid:
                    continue
                raise ValueError(f"Invalid sequence on line {row_number} in {path}") from None

            records.append(SequenceRecord(record_id=str(record_id), sequence=sequence))

    return records


def load_text_records(path: Path, skip_invalid: bool) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                sequence = normalize_sequence(stripped)
            except ValueError:
                if skip_invalid:
                    continue
                raise ValueError(f"Invalid sequence on line {line_number} in {path}") from None

            records.append(SequenceRecord(record_id=str(len(records)), sequence=sequence))

    return records


def iter_fasta_entries(path: Path) -> Iterable[tuple[str, str]]:
    current_header: str | None = None
    current_sequence: list[str] = []

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                if current_header is not None:
                    yield current_header, "".join(current_sequence)
                current_header = stripped[1:].strip() or f"record_{id(current_sequence)}"
                current_sequence = []
                continue
            current_sequence.append(stripped)

    if current_header is not None:
        yield current_header, "".join(current_sequence)


def load_fasta_records(path: Path, skip_invalid: bool) -> list[SequenceRecord]:
    records: list[SequenceRecord] = []
    for header, raw_sequence in iter_fasta_entries(path):
        try:
            sequence = normalize_sequence(raw_sequence)
        except ValueError:
            if skip_invalid:
                continue
            raise ValueError(f"Invalid sequence in FASTA record {header!r} from {path}") from None
        records.append(SequenceRecord(record_id=header, sequence=sequence))
    return records


def load_records(
    path: Path,
    input_format: str,
    sequence_column: str,
    id_column: str,
    skip_invalid: bool,
) -> list[SequenceRecord]:
    if input_format == "csv":
        return load_delimited_records(path, ",", sequence_column, id_column, skip_invalid)
    if input_format == "tsv":
        return load_delimited_records(path, "\t", sequence_column, id_column, skip_invalid)
    if input_format == "text":
        return load_text_records(path, skip_invalid)
    if input_format == "fasta":
        return load_fasta_records(path, skip_invalid)
    raise ValueError(f"Unsupported input format: {input_format}")


def to_kmers(sequence: str, kmer_size: int) -> str:
    if kmer_size <= 0:
        return sequence
    if len(sequence) < kmer_size:
        raise ValueError(
            f"Sequence length {len(sequence)} is shorter than k-mer size {kmer_size}"
        )
    return " ".join(sequence[index : index + kmer_size] for index in range(len(sequence) - kmer_size + 1))


def extract_kmer_size_from_text(text: str) -> int | None:
    for pattern in KMER_PATTERNS:
        match = pattern.search(text)
        if match:
            return int(match.group(1))
    return None


def infer_model_kmer_size(model_name: str) -> tuple[int | None, str]:
    direct_match = extract_kmer_size_from_text(model_name)
    if direct_match is not None:
        return direct_match, f"model name {model_name!r}"

    model_path = Path(model_name).expanduser()
    if not model_path.exists():
        return None, ""

    path_match = extract_kmer_size_from_text(model_path.name)
    if path_match is not None:
        return path_match, f"path name {model_path.name!r}"

    if model_path.is_dir():
        candidate_files = (
            model_path / "config.json",
            model_path / "tokenizer_config.json",
            model_path / "README.md",
            model_path / "model_index.json",
        )
        for candidate in candidate_files:
            if not candidate.is_file():
                continue
            text = candidate.read_text(encoding="utf-8", errors="ignore")
            match = extract_kmer_size_from_text(text)
            if match is not None:
                return match, str(candidate)

    return None, ""


def resolve_model_kmer_size(model_name: str, explicit_model_kmer_size: int | None) -> tuple[int, str]:
    if explicit_model_kmer_size is not None:
        return explicit_model_kmer_size, "--model-kmer-size"

    inferred_kmer_size, source = infer_model_kmer_size(model_name)
    if inferred_kmer_size is None:
        raise ValueError(
            "Could not infer the model's k-mer size from the model name or local metadata. "
            "Pass --model-kmer-size explicitly so the script can enforce compatibility."
        )
    return inferred_kmer_size, source


def validate_model_compatibility(
    requested_kmer_size: int,
    model_kmer_size: int,
    model_kmer_source: str,
) -> None:
    if requested_kmer_size == model_kmer_size:
        return

    message = (
        f"K-mer mismatch: requested --kmer-size {requested_kmer_size}, but the model expects "
        f"{model_kmer_size}-mers (detected from {model_kmer_source}). "
        f"Use a checkpoint trained on {requested_kmer_size}-mers instead."
    )
    if requested_kmer_size == 2:
        message += (
            " For the 2-mer side quest, you will need a real 2-mer DNABERT-style model; "
            "a 6-mer checkpoint will not give a meaningful comparison."
        )
    raise ValueError(message)


def resolve_output_path(input_path: Path, explicit_output: Path | None) -> Path:
    if explicit_output is not None:
        return explicit_output
    return input_path.with_name(f"{input_path.stem}_dnabert_embeddings.npz")


def import_embedding_dependencies():
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "This script requires the 'torch' and 'transformers' packages. "
            "Install them in the project environment before generating embeddings."
        ) from exc

    return torch, AutoModel, AutoTokenizer


def resolve_device(requested_device: str, torch) -> "torch.device":
    if requested_device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")
    if requested_device == "mps":
        has_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not has_mps:
            raise RuntimeError("MPS was requested, but no MPS device is available.")

    return torch.device(requested_device)


def pool_hidden_states(last_hidden_state, attention_mask, pooling: str, torch):
    if pooling == "cls":
        return last_hidden_state[:, 0, :]

    mask = attention_mask.unsqueeze(-1)

    if pooling == "mean":
        masked_hidden = last_hidden_state * mask
        token_counts = mask.sum(dim=1).clamp(min=1)
        return masked_hidden.sum(dim=1) / token_counts

    if pooling == "max":
        expanded_mask = attention_mask.unsqueeze(-1).bool()
        masked_hidden = last_hidden_state.masked_fill(~expanded_mask, torch.finfo(last_hidden_state.dtype).min)
        pooled = masked_hidden.max(dim=1).values
        pooled[~attention_mask.any(dim=1)] = 0
        return pooled

    raise ValueError(f"Unsupported pooling method: {pooling}")


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def generate_embeddings(
    records: list[SequenceRecord],
    model_name: str,
    batch_size: int,
    pooling: str,
    kmer_size: int,
    max_length: int,
    requested_device: str,
    trust_remote_code: bool,
) -> np.ndarray:
    torch, AutoModel, AutoTokenizer = import_embedding_dependencies()

    device = resolve_device(requested_device, torch)
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model.to(device)
    model.eval()

    if tokenizer.pad_token is None:
        fallback_token = tokenizer.eos_token or tokenizer.sep_token or tokenizer.unk_token
        if fallback_token is None:
            raise RuntimeError("Tokenizer does not define a pad token or usable fallback token.")
        tokenizer.pad_token = fallback_token
        if getattr(model.config, "pad_token_id", None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

    prepared_sequences = [to_kmers(record.sequence, kmer_size) for record in records]
    batches: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in batched(prepared_sequences, batch_size):
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}

            outputs = model(**encoded)
            hidden_state = outputs.last_hidden_state
            pooled = pool_hidden_states(hidden_state, encoded["attention_mask"], pooling, torch)
            batches.append(pooled.cpu().numpy())

    return np.vstack(batches).astype(np.float32)


def save_embeddings_npz(
    output_path: Path,
    records: list[SequenceRecord],
    embeddings: np.ndarray,
    model_name: str,
    pooling: str,
    kmer_size: int,
    model_kmer_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ids=np.asarray([record.record_id for record in records], dtype=str),
        sequences=np.asarray([record.sequence for record in records], dtype=str),
        embeddings=embeddings,
        model_name=np.asarray([model_name], dtype=str),
        pooling=np.asarray([pooling], dtype=str),
        kmer_size=np.asarray([kmer_size], dtype=np.int32),
        model_kmer_size=np.asarray([model_kmer_size], dtype=np.int32),
    )


def save_embeddings_csv(output_path: Path, records: list[SequenceRecord], embeddings: np.ndarray) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = ["id", "sequence"] + [f"embedding_{index}" for index in range(embeddings.shape[1])]

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for record, vector in zip(records, embeddings, strict=True):
            writer.writerow([record.record_id, record.sequence, *vector.tolist()])


def main() -> None:
    args = parse_args()
    input_path = args.input_path.resolve()
    input_format = detect_input_format(input_path, args.input_format)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.kmer_size < 0:
        raise ValueError("--kmer-size must be zero or greater")
    if args.model_kmer_size is not None and args.model_kmer_size < 0:
        raise ValueError("--model-kmer-size must be zero or greater")
    if args.max_length <= 0:
        raise ValueError("--max-length must be greater than zero")

    model_kmer_size, model_kmer_source = resolve_model_kmer_size(
        args.model_name,
        args.model_kmer_size,
    )
    validate_model_compatibility(args.kmer_size, model_kmer_size, model_kmer_source)

    records = load_records(
        input_path,
        input_format,
        args.sequence_column,
        args.id_column,
        args.skip_invalid,
    )
    if not records:
        raise ValueError(f"No valid sequences were loaded from {input_path}")

    embeddings = generate_embeddings(
        records=records,
        model_name=args.model_name,
        batch_size=args.batch_size,
        pooling=args.pooling,
        kmer_size=args.kmer_size,
        max_length=args.max_length,
        requested_device=args.device,
        trust_remote_code=args.trust_remote_code,
    )

    output_path = resolve_output_path(input_path, args.output)
    save_embeddings_npz(
        output_path=output_path,
        records=records,
        embeddings=embeddings,
        model_name=args.model_name,
        pooling=args.pooling,
        kmer_size=args.kmer_size,
        model_kmer_size=model_kmer_size,
    )
    print(f"Saved embeddings for {len(records)} sequences to {output_path}")

    if args.csv_output is not None:
        csv_output = args.csv_output.resolve()
        save_embeddings_csv(csv_output, records, embeddings)
        print(f"Saved CSV export to {csv_output}")


if __name__ == "__main__":
    main()
