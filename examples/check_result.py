"""Check a public example against the pinned traditional/GLM result."""
import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path, help="Run directory containing results/ or prepared/")
    parser.add_argument("--expected", type=Path, required=True)
    args = parser.parse_args()
    expected = json.loads(args.expected.read_text())
    result = args.result / expected["directory"]
    summary = json.loads((result / "summary.json").read_text())
    for key, value in expected["summary"].items():
        if summary.get(key) != value:
            raise SystemExit(f"Summary mismatch for {key}: {summary.get(key)!r} != {value!r}")
    with (result / "annotations.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    if len(rows) != len(expected["proteins"]):
        raise SystemExit("Protein count differs")
    actual = {row["protein_id"]: row for row in rows}
    for protein in expected["proteins"]:
        row = actual.get(protein["protein_id"], {})
        for key, value in protein.items():
            if row.get(key) != value:
                raise SystemExit(f"Mismatch for {protein['protein_id']} / {key}")
    print("Example result matches the pinned expected counts and annotations.")


if __name__ == "__main__":
    main()
