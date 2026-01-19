
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
KNOWLEDGE_BASE_DIR = REPO_ROOT / "knowledge_base" / "clean"
MAP_PATH = REPO_ROOT / "knowledge_base" / "terms_map.json"

SYLLABLES = [
    "ar", "en", "thor", "vel", "ix", "ran", "dor", "mal", "kyn", "zor",
    "ul", "eth", "qua", "zar", "val", "ryn", "tul", "vor", "gai", "lun",
]



def normalize_source_name(filename: str) -> str:
    name = re.sub(r"\.(txt|text)$", "", filename, flags=re.IGNORECASE)
    return name.strip(" -_\t")


def generate_target_name(existing: Dict[str, str]) -> str:
    for _ in range(1000):
        parts = random.choices(SYLLABLES, k=random.choice([2, 3]))
        candidate = ("".join(parts)).capitalize()
        if candidate not in existing.values():
            return candidate
    suffix = random.randint(1000, 9999)
    return f"Nomen{suffix}"


def load_terms_map() -> Dict[str, str]:
    if MAP_PATH.exists():
        try:
            data = json.loads(MAP_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception:
            pass
    return {}


def save_terms_map(terms_map: Dict[str, str]) -> None:
    MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAP_PATH.write_text(
        json.dumps(terms_map, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
    )


def get_txt_files() -> List[Path]:
    if not KNOWLEDGE_BASE_DIR.exists():
        return []
    return sorted([
        p for p in KNOWLEDGE_BASE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".txt"
    ])


def apply_replacements_to_text(text: str, terms_map: Dict[str, str]) -> Tuple[str, int]:
    result = text
    total_replacements = 0
    
    for source in sorted(terms_map.keys(), key=len, reverse=True):
        target = terms_map[source]
        if not target or not isinstance(target, str):
            continue

        pattern = re.compile(r'\b' + re.escape(source) + r'\b', re.IGNORECASE)
        
        matches = pattern.findall(result)
        if matches:
            total_replacements += len(matches)
            def replace_match(m):
                matched = m.group(0)
                if matched[0].isupper():
                    return target[0].upper() + target[1:] if len(target) > 1 else target.upper()
                else:
                    return target.lower()
            result = pattern.sub(replace_match, result)
    
    return result, total_replacements


def apply_content_replacements(terms_map: Dict[str, str]) -> None:
    txt_files = get_txt_files()
    
    if not txt_files:
        print("No .txt files found to process.")
        return
    
    print("-" * 60)
    
    total_files_modified = 0
    total_replacements = 0
    
    for filepath in txt_files:
        try:
            original_content = filepath.read_text(encoding="utf-8")
            modified_content, replacement_count = apply_replacements_to_text(
                original_content, terms_map
            )
            
            if replacement_count > 0:
                total_files_modified += 1
                total_replacements += replacement_count
                print(f"  {filepath.name}: {replacement_count} replacements")

                filepath.write_text(modified_content, encoding="utf-8")
        except Exception as e:
            print(f"  Error processing {filepath.name}: {e}")
    
    print("-" * 60)
    print(f"Modified'{total_files_modified} files")
    print(f"Total replacements: {total_replacements}")


def get_new_filename(original_name: str, terms_map: Dict[str, str]) -> str | None:
    entity_name = normalize_source_name(original_name)
    
    if entity_name in terms_map:
        new_entity = terms_map[entity_name]
        return f"{new_entity}.txt"
    
    return None


def rename_files(terms_map: Dict[str, str]) -> None:
    txt_files = get_txt_files()
    
    if not txt_files:
        print("No .txt files found to rename.")
        return
    
    print("Renaming files...")
    print("-" * 60)
    
    renamed_count = 0
    rename_operations: List[Tuple[Path, Path]] = []
    
    for filepath in txt_files:
        new_name = get_new_filename(filepath.name, terms_map)
        if new_name and new_name != filepath.name:
            new_path = filepath.parent / new_name
            rename_operations.append((filepath, new_path))
    
    new_names = [op[1].name for op in rename_operations]
    if len(new_names) != len(set(new_names)):
        print("Warning: Detected naming conflicts. Some files would have the same name.")
        seen = set()
        duplicates = set()
        for name in new_names:
            if name in seen:
                duplicates.add(name)
            seen.add(name)
        for dup in duplicates:
            print(f"  Conflict: {dup}")
    
    for old_path, new_path in rename_operations:
        print(f"  {old_path.name} -> {new_path.name}")
        renamed_count += 1

        try:
            if new_path.exists():
                print(f"    Warning: {new_path.name} already exists, skipping")
                continue
            old_path.rename(new_path)
        except Exception as e:
            print(f"    Error: {e}")
    
    print("-" * 60)
    print(f"Renamed {renamed_count} files")


def build_terms_map(seed: int | None = None) -> Dict[str, str]:
    if seed is not None:
        random.seed(seed)
    
    existing = load_terms_map()
    txt_files = get_txt_files()
    
    if not txt_files:
        print(f"No .txt files found in {KNOWLEDGE_BASE_DIR}")
        return existing
    
    updated = dict(existing)
    added_count = 0
    
    for path in txt_files:
        src_name = normalize_source_name(path.name)
        if src_name not in updated:
            updated[src_name] = generate_target_name(updated)
            added_count += 1
    
    save_terms_map(updated)
    print(f"Terms map saved to: {MAP_PATH}")
    print(f"Total entries: {len(updated)} (added {added_count})")
    
    return updated


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build/update terms_map.json and optionally apply replacements",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--apply-content",
        action="store_true",
        help="Apply term replacements to file contents"
    )
    parser.add_argument(
        "--rename-files",
        action="store_true",
        help="Rename files according to terms map"
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Skip building/updating the terms map (use existing)"
    )
    
    args = parser.parse_args()
    
    # Build or load terms map
    if args.skip_build:
        terms_map = load_terms_map()
        if not terms_map:
            print("Error: No existing terms map found and --skip-build was specified")
            return
        print(f"Using existing terms map with {len(terms_map)} entries")
    else:
        terms_map = build_terms_map(42)
    
    # Apply content replacements if requested
    if args.apply_content:
        apply_content_replacements(terms_map)
    
    # Rename files if requested
    if args.rename_files:
        rename_files(terms_map)

if __name__ == "__main__":
    main()
