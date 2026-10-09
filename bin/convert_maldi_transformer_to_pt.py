import h5torch
import torch
from tqdm import tqdm
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Convert MALDI Transformer data to PyTorch format.")
    parser.add_argument('--input_h5torch_path', type=str, required=True,
                        help='Path to the input h5torch file.')
    parser.add_argument('--output_pt_dir', type=str, required=True,
                        help='Path to the output directory.')
    args = parser.parse_args()
    
    input_h5torch_path = Path(args.input_h5torch_path)
    if not input_h5torch_path.exists():
        raise FileNotFoundError(f"Input path {input_h5torch_path} does not exist.")

    output_pt_dir = Path(args.output_pt_dir)
    if not output_pt_dir.exists():
        output_pt_dir.mkdir(parents=True, exist_ok=True)

    # Load the h5torch file
    dataset = h5torch.Dataset(str(input_h5torch_path))

    for x in tqdm(dataset):
        loc_value = x["0/loc"]
        # Handle both bytes and string formats
        if isinstance(loc_value, bytes):
            loc_str = loc_value.decode('utf-8')
        else:
            loc_str = str(loc_value)
        # Remove any leading slashes
        id = Path(loc_str.lstrip('/')).stem

        mz_array = torch.tensor(x["0/mz"])
        intensity_array = torch.tensor(x["0/intensity"])
        assert len(mz_array) == len(intensity_array), "Length of mz and intensity arrays must match."

        if len(mz_array) == 0:
            continue

        stacked_array = torch.stack((mz_array, intensity_array), dim=-1)

        torch.save(stacked_array, output_pt_dir / f"{id}.pt")


    
if __name__ == "__main__":
    main()