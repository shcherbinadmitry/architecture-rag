#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import json
import re

REPO_ROOT = Path(__file__).resolve().parents[1]
CLEAN_DIR = REPO_ROOT / "knowledge_base"
DEFAULT_PERSIST = REPO_ROOT / "indexing"
DEFAULT_COLLECTION = "knowledge_base"
TERMS_MAP_PATH = REPO_ROOT / "knowledge_base" / "terms_map.json"


@dataclass
class Chunk:
    doc_id: str
    chunk_id: int
    text: str
    start_char: int
    end_char: int
    source_path: str
    title: str


def read_text_files(root: Path) -> List[Tuple[str, str]]:
    files = sorted([p for p in root.glob("**/*.txt") if p.is_file()])
    items: List[Tuple[str, str]] = []
    for p in files:
        try:
            items.append((str(p), p.read_text(encoding="utf-8")))
        except Exception:
            continue
    return items


def recursive_split(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[Tuple[str, int, int]]:
    import re
    text_norm = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks: List[Tuple[str, int, int]] = []
    i = 0
    L = len(text_norm)
    while i < L:
        end = min(i + chunk_size, L)
        window = text_norm[i:end]
        if end < L:
            lookback = min(120, len(window))
            lb_seg = window[-lookback:]
            m = re.search(r"[\.!?]\s+|\n+|\s{2,}", lb_seg[::-1])
            if m:
                cut = len(window) - m.start()
                end = i + cut
                window = text_norm[i:end]
        chunks.append((window, i, end))
        if end == L:
            break
        i = max(end - chunk_overlap, 0)
        if i >= L:
            break
    return chunks


def apply_terms_map(text: str, mapping: Dict[str, str]) -> str:
    if not mapping:
        return text
    result = text
    for key in sorted(mapping.keys(), key=len, reverse=True):
        val = mapping.get(key)
        if not val or not isinstance(val, str):
            continue

        pattern = re.escape(key)
        result = re.sub(pattern, val, result)
    return result


def make_chunks(files: List[Tuple[str, str]], chunk_size: int, chunk_overlap: int, terms_map: Dict[str, str] | None = None) -> List[Chunk]:
    results: List[Chunk] = []
    for path, content in files:
        filepath = Path(path).resolve()
        title = filepath.stem
        doc_id = title

        try:
            source_path = str(filepath.relative_to(REPO_ROOT))
        except ValueError:
            source_path = filepath.name
        for idx, (chunk_text, s, e) in enumerate(recursive_split(content, chunk_size, chunk_overlap)):
            if terms_map:
                chunk_text = apply_terms_map(chunk_text, terms_map)
            results.append(Chunk(
                doc_id=doc_id,
                chunk_id=idx,
                text=chunk_text,
                start_char=s,
                end_char=e,
                source_path=source_path,
                title=title,
            ))
    return results


def embed_texts(model_name: str, texts: List[str], batch_size: int = 64) -> np.ndarray:
    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    return vectors.astype("float32")


def ensure_collection(client: chromadb.ClientAPI, name: str) -> chromadb.api.models.Collection.Collection:
    try:
        coll = client.get_collection(name)
    except Exception:
        coll = client.create_collection(name)
    return coll


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ChromaDB index from cleaned KB")
    parser.add_argument("--clean-dir", default=str(CLEAN_DIR), help="Path to cleaned .txt files root")
    parser.add_argument("--persist-dir", default=str(DEFAULT_PERSIST), help="Chroma persistence directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION, help="Chroma collection name")
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="SentenceTransformers model name")
    parser.add_argument("--chunk-size", type=int, default=800, help="Chunk size in characters")
    parser.add_argument("--chunk-overlap", type=int, default=100, help="Chunk overlap in characters")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--use-terms-map", action="store_true", help="Apply knowledge_base/terms_map.json replacements before embedding")
    args = parser.parse_args()

    clean_dir = Path(args.clean_dir)
    persist_dir = Path(args.persist_dir)

    if not clean_dir.exists():
        raise SystemExit(f"Clean directory does not exist: {clean_dir}")

    files = read_text_files(clean_dir)
    if not files:
        raise SystemExit(f"No .txt files found in {clean_dir}")

    # Load terms map if requested
    terms_map: Dict[str, str] = {}
    if args.use_terms_map:
        try:
            raw = TERMS_MAP_PATH.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict):
                terms_map = {str(k): str(v) for k, v in data.items() if isinstance(k, str)}
                print(f"Using terms_map.json with {len(terms_map)} entries")
            else:
                print("terms_map.json is not a JSON object; ignoring")
        except FileNotFoundError:
            print(f"terms_map.json not found at {TERMS_MAP_PATH}; proceeding without it")
        except Exception as e:
            print(f"Failed to load terms_map.json: {e}; proceeding without it")

    print(f"Chunking {len(files)} files...")
    chunks = make_chunks(files, args.chunk_size, args.chunk_overlap, terms_map if args.use_terms_map else None)
    print(f"Total chunks: {len(chunks)}")

    print(f"Embedding chunks with model '{args.model}' ...")
    embeddings = embed_texts(args.model, [c.text for c in chunks], batch_size=args.batch_size)

    print(f"Initializing Chroma at {persist_dir} ...")
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.Client(Settings(persist_directory=str(persist_dir), is_persistent=True))
    coll = ensure_collection(client, args.collection)

    print("Upserting into Chroma collection ...")
    B = 512
    for i in tqdm(range(0, len(chunks), B)):
        batch = chunks[i:i+B]
        vecs = embeddings[i:i+B].tolist()
        ids = [f"{c.doc_id}__{c.chunk_id}" for c in batch]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "chunk_id": c.chunk_id,
                "start_char": c.start_char,
                "end_char": c.end_char,
                "source_path": c.source_path,
                "title": c.title,
                "masked": bool(args.use_terms_map),
            }
            for c in batch
        ]
        documents = [c.text for c in batch]
        coll.upsert(ids=ids, embeddings=vecs, metadatas=metadatas, documents=documents)

    print("Chroma DB build finished (auto-persisted).")

if __name__ == "__main__":
    main()
