from models.autoencoder import Autoencoder
from datamodule import Spectrum_DataModule
from lightning.pytorch.loggers import TensorBoardLogger
import lightning as L
import torch

def main():

    hyperparameters = {
        'input_dim': 1800,
        'output_dim': 1800,
        'hidden_dim': 600,
        'bottleneck_dim': 300,
        'hidden_layers': 3,
        'weight_decay': 1e-5,
        'dropout': 0.2,
    }
    logger = TensorBoardLogger('lightning_logs', name='autoencoder_contrastive_model')

    model = Autoencoder(hyperparameters)

    # Load an autoencoder model
    # model = Autoencoder.load_from_checkpoint('./lightning_logs/autoencoder_model/version_15/checkpoints/epoch=149-step=3300.ckpt')
    # model = Autoencoder.load_from_checkpoint('./lightning_logs/autoencoder_model/version_21/checkpoints/epoch=149-step=11550.ckpt')
    print(model)
    model.convert() # TODO: disable backprop on first layers if needed
    
    torch.set_float32_matmul_precision('medium')    # medium | high
    
    datamodule = Spectrum_DataModule('../../data/idbac_db/preprocessing',
                                    '../../data/idbac_db/raw/ammended_db.csv',
                                    '../../data/idbac_db/processed_data',
                                    num_workers=7, 
                                    wipe_test_sets=False,)
                                    # wipe_test_sets=True)  #DEBUG
    datamodule.setup('fit')
    datamodule.plot(0)

    # Plot the train/test split
    # datamodule.full_dataset.plot_split('./train_test_split.png')

    trainer = L.Trainer(max_epochs=50, log_every_n_steps=10, logger=logger, devices=1)
    tuner = L.pytorch.tuner.Tuner(trainer)
    
    lr_find_results = tuner.lr_find(model,
                                    datamodule,
                                    min_lr=0.0001,   # 0.001
                                    max_lr=1.0,    # 1.0
                                    early_stop_threshold=None)
    model.lr = lr_find_results.suggestion()
    print("Best learning rate: ", model.lr)

    trainer.fit(model, datamodule)

    

if __name__=="__main__":
    main()