import os
import json
import argparse
from pathlib import Path

def merge_json_files(output_dir, merged_filename):
    merged_results = []
    chunked_files = sorted(
        [f for f in output_dir.glob("*.json") if "_" in f.stem and all(part.isdigit() for part in f.stem.split("_"))],
        key=lambda f: (int(f.stem.split('_')[0]), int(f.stem.split('_')[1]))
    )
    for file in chunked_files:
        with file.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                    merged_results.append(sample)
                except json.JSONDecodeError as e:
                    print(f"Skipping invalid JSON in {file}: {e}")
    merged_path = output_dir / merged_filename
    with merged_path.open('w', encoding='utf-8') as f:
        for sample in merged_results:
            f.write(json.dumps(sample, ensure_ascii=False))
            f.write('\n')
    return merged_path, merged_results

def evaluate_accuracy(samples):
    total = len(samples)
    correct = sum(1 for sample in samples if sample.get("acc") == "True")
    accuracy = correct / total if total > 0 else 0.0
    return total, correct, accuracy

def main():
    parser = argparse.ArgumentParser(description="Merge chunked JSON outputs and evaluate accuracy.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/nextqa"),
        help="Directory containing chunked JSON files."
    )
    parser.add_argument(
        "--merged-file", type=str, default="merged.json",
        help="Name of the merged JSON file to write."
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    if not output_dir.is_dir():
        print(f"Error: output directory '{output_dir}' does not exist.")
        return

    merged_path, samples = merge_json_files(output_dir, args.merged_file)
    total, correct, accuracy = evaluate_accuracy(samples)

    print(f"Merged {len(samples)} samples into '{merged_path}'.")
    print(f"Total samples: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    main()