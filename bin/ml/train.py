from models import MLP
# from dataset import MALDI_TOF_DS
from datamodule import Spectrum_DataModule
import lightning as L

def main():
    model = MLP(6000, 300, 250, 3)  # input_dim, output_dim, hidden_dim, hidden_layers
    print(model)

    # dataset = MALDI_TOF_DS('../../data/idbac_db/preprocessing',
    #                         '../../data/idbac_db/raw/db.csv',
    #                         '../../data/idbac_db/preprocessed',)
    
    datamodule = Spectrum_DataModule('../../data/idbac_db/preprocessing',
                                    '../../data/idbac_db/raw/db.csv',
                                    '../../data/idbac_db/preprocessed')

    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, datamodule)

    

if __name__=="__main__":
    main()