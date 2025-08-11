#!/usr/bin/env python3
"""
Download the full OpenSAT question bank and export as a single JSON and/or CSV.

Requirements:
  pip install requests

Examples:
  python opensat_dump.py --json questions.json
  python opensat_dump.py --csv questions.csv
  python opensat_dump.py --json questions.json --csv questions.csv
"""

import argparse
import csv
import json
import sys
from typing import Any, Dict, Iterable, List, Tuple
import requests

DEFAULT_SOURCE = "https://api.jsonsilo.com/public/942c3c3b-3a0c-4be3-81c2-12029def19f5"  # from OpenSAT repo

def fetch_bank(url: str) -> Dict[str, Any]:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    data = r.json()
    if not isinstance(data, dict):
        raise ValueError("Unexpected JSON shape (expected object at top level).")
    return data

def iter_questions(bank: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
    """
    OpenSAT stores questions grouped by section in the top-level dict.
    Each value is expected to be a list of question objects.
    We yield a normalized dict for each question.
    """
    for section, items in bank.items():
        if not isinstance(items, list):
            continue
        for q in items:
            # Expected shape per README example:
            # {
            #   "id": "...",
            #   "domain": "...",
            #   "question": {
            #       "paragraph": "...",
            #       "question": "...",
            #       "choices": {"A":"...","B":"...","C":"...","D":"..."},
            #       "correct_answer": "A",
            #       "explanation": "..."
            #   }
            # }
            qid = q.get("id")
            domain = q.get("domain")
            qobj = q.get("question", {}) or {}
            paragraph = qobj.get("paragraph")
            prompt = qobj.get("question")
            choices = qobj.get("choices") or {}
            correct_letter = qobj.get("correct_answer")
            explanation = qobj.get("explanation")

            # Map correct letter to its text if present
            correct_text = None
            if isinstance(choices, dict) and isinstance(correct_letter, str):
                correct_text = choices.get(correct_letter)

            # Flatten choices in stable letter order
            choice_pairs: List[Tuple[str, Any]] = []
            if isinstance(choices, dict):
                for k in sorted(choices.keys()):
                    choice_pairs.append((k, choices[k]))

            flat = {
                "id": qid,
                "section": section,       # group name from top-level
                "domain": domain,
                "paragraph": paragraph,
                "prompt": prompt,
                "correct_answer_letter": correct_letter,
                "correct_answer_text": correct_text,
                "explanation": explanation,
                "choices": choices,       # keep raw dict too
            }

            # Also put choices_A, choices_B, ... (handy for CSV)
            for letter, text in choice_pairs:
                flat[f"choice_{letter}"] = text

            yield flat

def write_json(rows: Iterable[Dict[str, Any]], out_path: str) -> None:
    data = list(rows)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def write_csv(rows: Iterable[Dict[str, Any]], out_path: str) -> None:
    rows = list(rows)
    # Build a union of keys so CSV has all columns
    fieldnames = set()
    for r in rows:
        fieldnames.update(r.keys())
    # Order a few important columns first
    preferred = [
        "id", "section", "domain", "paragraph", "prompt",
        "choice_A", "choice_B", "choice_C", "choice_D",
        "correct_answer_letter", "correct_answer_text", "explanation"
    ]
    ordered = preferred + sorted(k for k in fieldnames if k not in preferred)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=ordered)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Dump the OpenSAT question bank.")
    p.add_argument("--source", default=DEFAULT_SOURCE,
                   help="JSON endpoint to fetch (default: OpenSAT public JSON).")
    p.add_argument("--json", metavar="PATH", help="Write combined JSON to PATH.")
    p.add_argument("--csv", metavar="PATH", help="Write combined CSV to PATH.")
    args = p.parse_args(argv)

    if not args.json and not args.csv:
        p.error("Pick at least one output: --json and/or --csv")

    try:
        bank = fetch_bank(args.source)
    except Exception as e:
        print(f"Error fetching source JSON: {e}", file=sys.stderr)
        return 2

    # Prepare an iterator twice (for writing both json & csv)
    all_rows = list(iter_questions(bank))

    if args.json:
        try:
            write_json(all_rows, args.json)
            print(f"Wrote {len(all_rows)} questions to {args.json}")
        except Exception as e:
            print(f"Error writing JSON: {e}", file=sys.stderr)
            return 3

    if args.csv:
        try:
            write_csv(all_rows, args.csv)
            print(f"Wrote {len(all_rows)} questions to {args.csv}")
        except Exception as e:
            print(f"Error writing CSV: {e}", file=sys.stderr)
            return 4

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
