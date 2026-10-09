import os
import argparse
import torch
import numpy as np
from maldi_nn.models import AMRModel, MaldiTransformer
from maldi_nn.utils.data import SpeciesClfDataModule, DRIAMSSpectrumDataModule
from maldi_nn.spectrum import PeakFilter
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch import Trainer
import ast
from maldi_nn.reproduce.modules import (
    MaldiTransformerNegSampler,
    MaldiTransformerOnlyClf,
    MaldiTransformerMaskMSE,
)
from maldi_nn.utils.metrics import * 

import pandas as pd
from pathlib import Path
from tqdm import tqdm
import sys

def test(
    checkpoint_path,
    data_path,
    metadata_path=None
):

    model = MaldiTransformer.load_from_checkpoint(checkpoint_path)
    print(model)

    # They would use this
    # dm = SpeciesClfDataModule(
    #     data_path,
    #     batch_size=128,
    #     n_workers=12,
    #     preprocessor=(
    #         PeakFilter(max_number=200)
    #     ),
    #     in_memory=True,
    # )

    # We will try this because the train/val/test splits actually work
    if True:
        dm = DRIAMSSpectrumDataModule(
                data_path,
                batch_size=2,
                n_workers=12,
                preprocessor=PeakFilter(max_number=200),
                in_memory=True,
                exclude_nans=(True),
        )

        dm.setup(None)

        trainer = Trainer(
            accelerator="gpu",
            strategy="auto",
            max_epochs=1,
        )

        print("dm.test_dataloader()", len(dm.test_dataloader()))

        # Show the first batch
        for batch in dm.test_dataloader():
            print("Batch keys:", batch.keys())
            print("Batch shapes:")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k}: {v.shape}")
                    # Show first 5 values of the tensor
                    print(f"    {k} values: {v[:5]}")
                else:
                    print(f"  {k}: {type(v)}")
            break

        # test
            # test
        p = trainer.validate(
            model, dataloaders=dm.test_dataloader()
        )

        print(p)

    strain_name_to_accession = {}
    if metadata_path is not None:
        print("Loading metadata...", flush=True)
        metadata_df = pd.read_csv(metadata_path)
        metadata_df['strain_name'] = metadata_df['code'].apply(lambda x: str(x).split('/')[-1].split('.')[0])
        strain_name_to_accession = dict(zip(metadata_df['strain_name'], metadata_df['accession']))
        print("Metadata loaded.", flush=True)

    if True:
        dm = DRIAMSSpectrumDataModule(
                data_path,
                batch_size=1,
                n_workers=12,
                preprocessor=PeakFilter(max_number=200),
                in_memory=True,
                exclude_nans=(True),
        )

        dm.setup(None)

        """
        embedding_dict = {
                                    'accession':metadata['accession'], 
                                    'strain_name':metadata['strain_name'],
                                    'embedding': np.squeeze(embedding),
                                    'pred_class': pred_class,
                                }
        """

        print("Performing predictions and saving to a file", flush=True)
        embeddings = []

        count = 0

        with torch.no_grad():
            model.eval()
            for idx, (b) in enumerate(tqdm(dm.test_dataloader(), desc="Embedding spectra")):
                for k, v in b.items():
                    if isinstance(v, torch.Tensor):
                        b[k] = v.to(model.device, torch.float)

                # print(b)

                count += 1

                # if count > 10:
                #     sys.exit(0)

                embedding = model.embed_step(b, idx)
                file_name = b['loc']
                label_index = b['species']
                label_name = dm.species_mapping[label_index.item()]

                accession = strain_name_to_accession.get(str(file_name).split('/')[-1].split('.')[0], None)
                if accession:
                    accession = [accession]

                embeddings.append({
                    'strain_name': [str(file_name).split('/')[-1].split('.')[0]],
                    'embedding': embedding.squeeze().cpu().numpy(),
                    'pred_class': None,
                    'accession': accession,
                    'true_class': label_name,
                    'pred_class': None
                })

        output_path = Path(checkpoint_path).parent / 'embeddings.feather'
        print(f"Saving embeddings to {output_path} ...", flush=True)
        df = pd.DataFrame(embeddings)

        print(df.head())

        # Get all cols and dtypes
        for col in df.columns:
            print(f"{col}: {df[col].dtype}")

        df.to_feather(output_path)
        print("Done.", flush=True)



def main():
    parser = argparse.ArgumentParser(
        description="Evaluation script for Maldi Transformer.",
    )
    parser.add_argument("--metadata_path", type=str, metavar="metadata_path", help="path to metadata csv file used to map strain names back to accessions (optional).")
    parser.add_argument("--model_path", type=str, metavar="model_path", help="path to model checkpoint.")
    parser.add_argument("--data_path", type=str, metavar="data_path", help="path to h5torch file.")

    args = parser.parse_args()

    # Ensure paths exist
    if not os.path.exists(args.model_path):
        raise ValueError(f"Model path {args.model_path} does not exist.")
    if not os.path.exists(args.data_path):
        raise ValueError(f"Data path {args.data_path} does not exist.")
    
    for var, val in vars(args).items():
        print(f"{var}: {val}")

    test(
        checkpoint_path=args.model_path,
        data_path=args.data_path,
        metadata_path=args.metadata_path,
    )


if __name__ == '__main__':
    main()