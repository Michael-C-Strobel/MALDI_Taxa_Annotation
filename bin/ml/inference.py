from models import MLP, RawCosine
from datamodule import Spectrum_DataModule
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
import torch
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--metric_path", type=str, default="metrics")
    parser.add_argument("--inference_set", type=str, default="test")
    args = parser.parse_args()

    datamodule = Spectrum_DataModule('../../data/idbac_db/preprocessing',
                                    '../../data/idbac_db/raw/db.csv',
                                    '../../data/idbac_db/preprocessed',
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
        if len(model_path) == 0:
            raise FileNotFoundError(f"No model found in {model_dir}")
        if len(model_path) > 1:
            raise FileNotFoundError(f"Multiple models found in {model_dir}")
        model_path = model_path[0]
        print(f"Loding model from {model_path}")

        # Load the model
        model = MLP.load_from_checkpoint(model_path)

    model.eval()
    logger = TensorBoardLogger('lightning_logs', name=str(args.model_name)+'/prediction')

    # Get the predictions
    predictions = []
    true_similarity = []

    trainer = L.Trainer(logger=logger)

    # Use the Trainer to run predictions
    outputs = trainer.predict(model, datamodule=datamodule, return_predictions=True)

    # Extract predictions and ground truth values
    predictions = []
    true_similarity = []

    for output in outputs:
        predictions.extend(output['predictions'].cpu().numpy())  # Assuming you want numpy arrays
        true_similarity.extend(output['similarity'].cpu().numpy())

    # Save the predictions
    predictions = np.array(predictions)
    true_similarity = np.array(true_similarity)

    metric_path = Path('lightning_logs') / str(args.model_name) / str(args.metric_path)
    if not metric_path.exists():
        metric_path.mkdir(parents=True, exist_ok=True)
    print("Saving metrics to", metric_path)

    np.save(metric_path / "preds.npy", predictions)
    np.save(metric_path / "true.npy", true_similarity)

    # Correlation of predictions with true values
    correlation = np.corrcoef(predictions, true_similarity)[0, 1]
    mae = np.mean(np.abs(predictions - true_similarity))
    rmse = np.sqrt(np.mean((predictions - true_similarity) ** 2))
    print("Correlation of predictions with true values:", correlation)
    # Save to txt
    with open(metric_path / "metrics.txt", 'w') as f:
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


if __name__ == "__main__":
    main()