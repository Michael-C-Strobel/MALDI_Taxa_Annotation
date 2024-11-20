import argparse
from pathlib import Path
import json
from typing import List

def merge_json(json_files:List[Path], output_file:Path)->None:
    """
    Merge JSON files.
    
    Args:
        json_files (List[Path]): List of JSON files to merge
        output_file (Path): Output file path

    Returns:
        None
    """
    merged_json = []
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as f:
            merged_json.extend(json.load(f))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(merged_json, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Merge JSON files.')
    parser.add_argument('--json_files', type=str, help='Input JSON files', required=True)
    parser.add_argument('--output_file', type=str, help='Output file path', required=True)
    args = parser.parse_args()

    json_files = args.json_files.split(';')
    json_files = [Path(json_file) for json_file in json_files]

    for json_file in json_files:
        if not json_file.exists():
            raise ValueError(f'JSON file does not exist: {json_file}')
        
    output_file = Path(args.output_file)

    if not output_file.parent.exists():
        output_file.parent.mkdir(parents=True, exist_ok=True)

    merge_json(json_files, output_file)

if __name__ == "__main__":
    main()