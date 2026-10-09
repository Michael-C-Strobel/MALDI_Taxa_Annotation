"""List publication inference commands; execute only with --execute.

Uses the exact checkpoint inventory and original helper-compatible directories.
Run with the ML environment active. No model imports are needed to list commands.
"""
import argparse
import csv
from pathlib import Path
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def commands(python):
    with (ROOT / 'publication/checkpoints.csv').open() as handle:
        checkpoints = list(csv.DictReader(handle))
    datasets = ('DRIAMS-A', 'DRIAMS-B', 'DRIAMS-C', 'DRIAMS-D', 'RKI')
    for row in checkpoints:
        model, split, fold = row['model'], row['split'], row['fold']
        checkpoint = ROOT / row['path']
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        run = checkpoint.parent.parent
        for dataset in datasets:
            if split == 'genera' and dataset not in ('DRIAMS-A', 'RKI'):
                continue
            sets = ('train', 'test') if dataset == 'DRIAMS-A' else ('all',)
            for subset in sets:
                if model == 'CLIP_Transformer' and dataset == 'DRIAMS-A':
                    output = run / 'inference/genera' / split
                elif dataset == 'DRIAMS-A':
                    output = run / 'inference/DRIAMS-A/genera' / split
                else:
                    output = run / 'inference' / dataset / 'all'
                cmd = [python, 'bare_inference.py', '--model', model, '--dataset', dataset,
                       '--target', 'genera', '--split_type', split, '--inference_set', subset,
                       '--run_for_score', '--new_paths', '--checkpoint_path', str(checkpoint),
                       '--output_dir', str(output), '-k', fold]
                if model == 'MaldiTransformerWrapper':
                    cmd.append('--maldi_nn_preprocessing')
                    if dataset == 'DRIAMS-A':
                        spectra = ROOT / 'data/driams/processed_data/MaldiTransformer_Genus_Labels' / split / 'spectra/all'
                        cmd += ['--spectra_path', str(spectra)]
                yield cmd
    for split in ('species', 'genera'):
        bins = (1, 3, 5, 7, 10) if split == 'species' else (10,)
        for fold in range(6):
            for width in bins:
                for subset in ('train', 'test'):
                    output = ROOT / f'bin/ml/lightning_logs_DRIAMS_A_for_score/cosine_{width}/genera/{split}/k={fold}'
                    yield [python, 'bare_inference.py', '--model', f'cosine_{width}', '--dataset', 'DRIAMS-A',
                           '--target', 'genera', '--split_type', split, '--inference_set', subset,
                           '--run_for_score', '--new_paths', '--output_dir', str(output), '-k', str(fold)]
    for dataset in datasets[1:]:
        suffix = dataset.replace('DRIAMS-', 'DRIAMS_')
        for width in (1, 3, 5, 7, 10):
            output = ROOT / f'bin/ml/lightning_logs_{suffix}_for_score/cosine_{width}/all'
            yield [python, 'bare_inference.py', '--model', f'cosine_{width}', '--dataset', dataset,
                   '--target', 'genera', '--split_type', 'species', '--inference_set', 'all',
                   '--run_for_score', '--new_paths', '--output_dir', str(output)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python', default='python', help='Python from the active ML environment')
    parser.add_argument('--execute', action='store_true', help='Run inference; default only lists commands')
    args = parser.parse_args()
    for cmd in commands(args.python):
        print(shlex.join(cmd), flush=True)
        if args.execute:
            subprocess.run(cmd, cwd=ROOT / 'bin/ml', check=True)


if __name__ == '__main__':
    main()
