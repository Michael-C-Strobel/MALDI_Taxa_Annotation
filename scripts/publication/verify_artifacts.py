"""Verify the frozen publication artifact hashes without running any analysis."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    root = args.repo.resolve()
    artifacts = json.loads((root / 'publication/frozen_artifacts.json').read_text())
    failures = []
    for record in artifacts:
        path = root / record['path']
        if not path.is_file():
            failures.append(f"Missing: {record['path']}")
            continue
        if path.stat().st_size != record['bytes']:
            failures.append(f"Size differs: {record['path']}")
            continue
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for chunk in iter(lambda: handle.read(8 * 1024**2), b''):
                digest.update(chunk)
        if digest.hexdigest() != record['sha256']:
            failures.append(f"Hash differs: {record['path']}")
    if failures:
        raise SystemExit('\n'.join(failures))
    print(f'Verified {len(artifacts)} frozen publication artifacts.')


if __name__ == '__main__':
    main()
