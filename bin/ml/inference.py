import argparse
from pathlib import Path
from typing import List, Tuple

from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

from models.mlp import MLP
from models.autoencoder import Autoencoder
from models.cosine import RawCosine

from datamodule import Spectrum_DataModule
from utils import mirror_plot

from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
from sklearn.metrics import fowlkes_mallows_score


def compute_fowlkes_mallows_score(y_true, y_pred, figure_path:Path=None, method:str='complete'):
    """ Compute the fowlkes_mallows_score.

    Parameters
    ----------
    y_true : np.ndarray
        The true similarity matrix as a condensed similarity matrix
    y_pred : np.ndarray
        The predicted similarity matrix as a condensed similarity matrix
    figure_path : Path, optional
        The path to save the dendrogram comparison plot, by default None
    method : str, optional
        The method to use for clustering, by default 'complete'

    Returns
    ----------
    
    """

    # Cluster
    y_true = 1 - np.array(y_true)
    y_pred = 1- np.array(y_pred)
    true_linkage = linkage(y_true, method=method)
    pred_linkage = linkage(y_pred, method=method)

    max_k = len(pred_linkage)

    scores = []
    for k in tqdm(list(range(2, max_k))):
        tc = np.squeeze(cut_tree(true_linkage, k))
        pc = np.squeeze(cut_tree(pred_linkage, k))
        scores.append(fowlkes_mallows_score(tc, pc))


    if figure_path:
        figure_path = Path(figure_path)
        if not figure_path.parent.exists():
            figure_path.parent.mkdir(parents=True, exist_ok=True)

        fig, axs = plt.subplots(1, 3, figsize=(15, 7))

        # Plot the dendrograms
        dendrogram(true_linkage, ax=axs[0], color_threshold=0.7 * max(true_linkage[:, 2]))
        # axs[0].set_title(f"True Linkage\nCophenetic Correlation: {true_cophenetic_corr:.2f}")
        axs[0].set_xlabel("Sample index")
        axs[0].set_ylabel("Distance")
        axs[0].set_title(f"True Linkage")

        dendrogram(pred_linkage, ax=axs[1], color_threshold=0.7 * max(pred_linkage[:, 2]))
        # axs[1].set_title(f"Predicted Linkage\nCophenetic Correlation: {pred_cophenetic_corr:.2f}")
        axs[1].set_xlabel("Sample index")
        axs[1].set_ylabel("Distance")
        axs[1].set_title(f"Predicted Linkage")

        # Plot the Fowlkes Mallows Score
        auc = np.trapz(scores, range(2, max_k))/max_k
        axs[2].plot(range(2, max_k), scores)
        axs[2].set_xlabel("Number of Clusters")
        axs[2].set_ylabel("Fowlkes Mallows Score")
        axs[2].set_title(f"Fowlkeys Mallows Score (Method={method})\nAUC={auc:.2f}")
        axs[2].set_ylim(-0.1, 1.1)

        plt.suptitle(f"Fowlkeys Mallows Score (Method={method})")
        plt.tight_layout()

        # Add some left, right margin
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.savefig(figure_path)
        plt.close(fig)

    return scores

def create_report(predictions:List[float], true_similarity:List[float], metadata:List[dict], output_path:Path, k:int=5):
    """Output the k worst predictions, with associated metadata to a folder.
    Spectra are saved as images and numpy arrays, metadata is saved as a json file.

    Parameters
    ----------
    predictions : List[float]
        The predicted similarities
    true_similarity : List[float]
        The true similarities
    metadata : List[dict]
        The metadata associated with the spectra
    output_path : Path
        The path to save the output to

    Returns
    ----------
    None
    """
    # Sort the predictions
    worst_predictions = np.argsort(np.abs(np.array(predictions) - np.array(true_similarity)))[::-1][:k]

    print("len(metadata)", len(metadata))
    print("len(predictions)", len(predictions))
    print("len(true_similarity)", len(true_similarity))

    for i, idx in enumerate(worst_predictions):
        subdir_path = output_path / f"worst_{i}"
        if not subdir_path.exists():
            subdir_path.mkdir(parents=True, exist_ok=True)

        print('idx', idx)
        # Save the spectra
        spectrum_a = metadata[idx]['spectrum_a']
        spectrum_b = metadata[idx]['spectrum_b']
        np.save(subdir_path / f"spectrum_a.npy", spectrum_a)
        np.save(subdir_path / f"spectrum_b.npy", spectrum_b)
        mirror_plot(spectrum_a, spectrum_b, subdir_path / f"mirror.png", title=f"True Similarity: {true_similarity[idx]:.2f}\nPredicted Similarity: {predictions[idx]:.2f}",
                    x_label="Mass Bin", y_label="Intensity")

        # Save the metadata
        output_metadata = {} #metadata[idx].copy()
        with open(subdir_path / f"metadata.json", "w", encoding="utf-8") as f:
            output_metadata['true_similarity'] = true_similarity[idx].item()
            output_metadata['predicted_similarity'] = predictions[idx].item()
            output_metadata['index'] = idx.item()
            output_metadata['error'] = np.abs(predictions[idx].item() - true_similarity[idx].item())
            print(output_metadata)
            json.dump(output_metadata, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--metric_path", type=str, default="metrics")
    parser.add_argument("--inference_set", type=str, default="test", choices=["test", "val", "train"])
    args = parser.parse_args()

    datamodule = Spectrum_DataModule('../../data/idbac_db/preprocessing',
                                    '../../data/idbac_db/raw/ammended_db.csv',
                                    '../../data/idbac_db/processed_data',
                                    num_workers=7, 
                                    wipe_test_sets=False,
                                    inference_set_to_use=args.inference_set)

    # Perform inference on all data
    datamodule.setup('test')

    if str(args.model_name).lower() == 'cosine':
        model = RawCosine()

    else:
        model_dir = Path('lightning_logs') / str(args.model_name) / 'checkpoints'
        model_path = list(model_dir.glob('*.ckpt'))
        # hparams = Path('lightning_logs') / str(args.model_name) / 'hparams.yaml'
        # hparams = torch.load(hparams)
        if len(model_path) == 0:
            raise FileNotFoundError(f"No model found in {model_dir}")
        if len(model_path) > 1:
            raise FileNotFoundError(f"Multiple models found in {model_dir}")
        model_path = model_path[0]
        print(f"Loding model from {model_path}")

        # Load the model
        if 'mlp' in args.model_name.lower():
            model = MLP.load_from_checkpoint(model_path)
        elif 'autoencoder' in args.model_name.lower():
            hyperparameters = { # Temporary fix until I figure out how to deal with this
                    'input_dim': 1800,
                    'output_dim': 1800,
                    'hidden_dim': 600,
                    'bottleneck_dim': 300,
                    'hidden_layers': 3,
                    'weight_decay': 1e-5,
                    'dropout': 0.2,
                }
            model = Autoencoder(hyperparameters)
            model.load_converted_from_checkpoint(model_path)
        else:
            raise ValueError(f"Unknown model name {args.model_name}")

    model.eval()
    logger = TensorBoardLogger('lightning_logs', name=str(args.model_name)+'/prediction')

    # Get the predictions
    trainer = L.Trainer(logger=logger, devices=1)

    # Use the Trainer to run predictions
    outputs = trainer.predict(model, datamodule=datamodule, return_predictions=True)

    # Extract predictions and ground truth values
    predictions = []
    true_similarity = []
    metadata = []

    for output in outputs:
        predictions.extend(output['predictions'].cpu().numpy())  # Assuming you want numpy arrays
        true_similarity.extend(output['similarity'].cpu().numpy())
        # output['metadata'] is a dict of lists
        # want to convert it to a list of dicts
        for i in range(len(output['metadata']['accession_a'])):
            metadata.append({
                k: v[i] for k, v in output['metadata'].items()
            })

    # Save the predictions
    predictions = np.array(predictions)
    true_similarity = np.array(true_similarity)
    if np.max(predictions) > 1 + 1e-6 or np.min(predictions) < 0 - 1e-6:
        raise ValueError("Predictions are not in the range [0, 1]")
    # Clamp (to handle numerical errors)
    predictions = np.clip(predictions, 0, 1)


    metric_path = Path('lightning_logs') / str(args.model_name) / str(args.metric_path)
    if not metric_path.exists():
        metric_path.mkdir(parents=True, exist_ok=True)
    print("Saving metrics to", metric_path)

    # print(metadata)
    # print(len(metadata))

    np.save(metric_path / "preds.npy", predictions)
    np.save(metric_path / "true.npy", true_similarity)
    # Save metadata (list of dicts)
    with open(metric_path / "prediction_metadata.json", "w", encoding="utf-8") as f:
        # Change all tensors to lists for json serialization
        for i in range(len(metadata)):
            for k, v in metadata[i].items():
                if isinstance(v, np.ndarray):
                    metadata[i][k] = v.tolist()
        # json.dump(metadata, f, indent=4)

    # Create a report of the worst predictions
    create_report(predictions, true_similarity, metadata, metric_path / "worst_predictions", k=5)

    f = open(metric_path / "metrics.txt", 'w', encoding='utf-8')

    # Fowlkes Mallows Score
    scores = compute_fowlkes_mallows_score(true_similarity, predictions, metric_path / "dendrograms.png")
    print("Fowlkes Mallows Score:", scores)
    f.write("Fowlkes Mallows Score: \n")
    f.write(str(scores) + "\n")

    # Correlation of predictions with true values
    correlation = np.corrcoef(predictions, true_similarity)[0, 1]
    mae = np.mean(np.abs(predictions - true_similarity))
    rmse = np.sqrt(np.mean((predictions - true_similarity) ** 2))
    print("Correlation of predictions with true values:", correlation)
    # Save to txt
    f.write(f"Correlation of predictions with true values: R2={correlation:.2f} \n")
    f.write(f"Mean Absolute Error: {mae:.2f} \n")
    f.write(f"Root Mean Squared Error: {rmse:.2f} \n")
    # Plot
    fig = plt.figure()
    sns.scatterplot(x=true_similarity, y=predictions, alpha=0.5)
    x_min = min(min(true_similarity), min(predictions))
    x_max = max(max(true_similarity), max(predictions))
    y_min = min(min(true_similarity), min(predictions))
    y_max = max(max(true_similarity), max(predictions))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("True Sequence Similarity Similarity")
    plt.ylabel("Predicted Sequence Similarity")
    plt.title(f"Predicted vs True similarity (R2={correlation:.2f})")
    plt.savefig(metric_path / "scatter.png")

    # Hex Density Plot
    hexplot = sns.jointplot(x=true_similarity, y=predictions, kind='hex', xlim=(x_min, x_max), ylim=(y_min, y_max))
    plt.subplots_adjust(left=0.2, right=0.8, top=0.8, bottom=0.2)
    cbar_ax = hexplot.figure.add_axes([.85, .25, .05, .4])  # x, y, width, height
    hexplot.set_axis_labels("True Sequence Similarity Similarity", "Predicted Sequence Similarity")
    plt.colorbar(cax=cbar_ax)
    plt.suptitle(f"Predicted vs True similarity (R2={correlation:.2f})")
    # plt.tight_layout()
    plt.savefig(metric_path / "hex.png")

    f.close()


if __name__ == "__main__":
    main()