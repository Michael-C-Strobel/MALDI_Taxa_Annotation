from models import MLP
from dataset import MALDI_TOF_DS
import lightning as L

def main():
    model = MLP(28*28, 10, 128, 3)  # input_dim, output_dim, hidden_dim, hidden_layers
    print(model)

    dataset = MALDI_TOF_DS()

    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, dataset)

    

if __name__=="__main__":
    main()