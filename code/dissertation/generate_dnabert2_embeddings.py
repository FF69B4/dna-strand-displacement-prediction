from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dissertation.dnabert2_backend.bert_layers as dnabert2_layers
from dissertation.dnabert2_backend import BertConfig, BertModel
from dissertation.generate_dna_bert_embeddings import (
    SequenceRecord,
    detect_input_format,
    load_records,
    normalize_sequence,
)


MODEL_NAME = "zhihan1996/DNABERT-2-117M"


@dataclass(frozen=True)
class Dnabert2EmbeddingConfig:
    model_name: str
    pooling: str
    batch_size: int
    max_length: int
    device: str
    allow_overwrite: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate DNABERT-2 embeddings using the dissertation-local backend. "
            "Outputs are separate from the existing 6-mer DNABERT artifacts."
        )
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    parser.add_argument("--input-format", choices=("auto", "csv", "tsv", "text", "fasta"), default="auto")
    parser.add_argument("--sequence-column", default="Sequence")
    parser.add_argument("--id-column", default="No.")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--pooling", choices=("cls", "mean", "max"), default="mean")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--skip-invalid", action="store_true")
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Allow replacing an existing output file. Omitted by default to protect current work.",
    )
    return parser.parse_args()


def refuse_overwrite(path: Path, allow_overwrite: bool) -> None:
    if path.exists() and not allow_overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing file: {path}. "
            "Pass --allow-overwrite only if replacing it is intentional."
        )


def resolve_device(requested_device: str, torch):
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
            raise RuntimeError("MPS was requested, but not available.")
    return torch.device(requested_device)


def batched(items: list[str], batch_size: int) -> Iterable[list[str]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def pool_hidden_states(last_hidden_state, attention_mask, pooling: str, torch):
    if pooling == "cls":
        return last_hidden_state[:, 0, :]
    mask = attention_mask.unsqueeze(-1)
    if pooling == "mean":
        token_counts = mask.sum(dim=1).clamp(min=1)
        return (last_hidden_state * mask).sum(dim=1) / token_counts
    if pooling == "max":
        expanded_mask = attention_mask.unsqueeze(-1).bool()
        masked = last_hidden_state.masked_fill(~expanded_mask, torch.finfo(last_hidden_state.dtype).min)
        pooled = masked.max(dim=1).values
        pooled[~attention_mask.any(dim=1)] = 0
        return pooled
    raise ValueError(f"Unsupported pooling method: {pooling}")


def load_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)


def load_model(model_name: str, tokenizer):
    import torch
    from huggingface_hub import hf_hub_download

    dnabert2_layers.flash_attn_qkvpacked_func = None
    config = BertConfig.from_pretrained(model_name)
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    model = BertModel(config, add_pooling_layer=False)
    checkpoint_path = hf_hub_download(model_name, "pytorch_model.bin")
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict, strict=False)
    return model


def extract_last_hidden_state(outputs):
    if hasattr(outputs, "last_hidden_state"):
        return outputs.last_hidden_state
    if isinstance(outputs, tuple):
        encoded_layers = outputs[0]
        if isinstance(encoded_layers, list):
            return encoded_layers[-1]
        return encoded_layers
    raise TypeError(f"Unsupported DNABERT-2 output type: {type(outputs)!r}")


def generate_embeddings(
    records: list[SequenceRecord],
    config: Dnabert2EmbeddingConfig,
) -> np.ndarray:
    import torch

    device = resolve_device(config.device, torch)
    tokenizer = load_tokenizer(config.model_name)
    model = load_model(config.model_name, tokenizer)
    model.to(device)
    model.eval()

    sequences = [normalize_sequence(record.sequence) for record in records]
    embeddings: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in batched(sequences, config.batch_size):
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            outputs = model(**encoded)
            hidden_state = extract_last_hidden_state(outputs)
            pooled = pool_hidden_states(hidden_state, encoded["attention_mask"], config.pooling, torch)
            embeddings.append(pooled.detach().cpu().numpy())

    return np.vstack(embeddings).astype(np.float32)


def save_embeddings_npz(
    output_path: Path,
    records: list[SequenceRecord],
    embeddings: np.ndarray,
    config: Dnabert2EmbeddingConfig,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        ids=np.asarray([record.record_id for record in records], dtype=str),
        sequences=np.asarray([record.sequence for record in records], dtype=str),
        embeddings=embeddings,
        model_name=np.asarray([config.model_name], dtype=str),
        pooling=np.asarray([config.pooling], dtype=str),
        kmer_size=np.asarray([0], dtype=np.int32),
        model_kmer_size=np.asarray([0], dtype=np.int32),
        tokenizer=np.asarray(["DNABERT-2 BPE"], dtype=str),
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
    output_path = args.output.resolve()
    csv_output = args.csv_output.resolve() if args.csv_output is not None else None

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be greater than zero")
    if args.max_length <= 0:
        raise ValueError("--max-length must be greater than zero")
    refuse_overwrite(output_path, args.allow_overwrite)
    if csv_output is not None:
        refuse_overwrite(csv_output, args.allow_overwrite)

    records = load_records(
        input_path,
        detect_input_format(input_path, args.input_format),
        args.sequence_column,
        args.id_column,
        args.skip_invalid,
    )
    if not records:
        raise ValueError(f"No valid sequences were loaded from {input_path}")

    config = Dnabert2EmbeddingConfig(
        model_name=args.model_name,
        pooling=args.pooling,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        allow_overwrite=args.allow_overwrite,
    )
    embeddings = generate_embeddings(records, config)
    save_embeddings_npz(output_path, records, embeddings, config)
    print(f"Saved DNABERT-2 embeddings for {len(records)} sequences to {output_path}")

    if csv_output is not None:
        save_embeddings_csv(csv_output, records, embeddings)
        print(f"Saved CSV export to {csv_output}")


if __name__ == "__main__":
    main()
