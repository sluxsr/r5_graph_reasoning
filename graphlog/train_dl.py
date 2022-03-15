from dataloader import *

class TrainLoader:
    def __init__(self,hparams):
        self.hparams = hparams
        self.data_dir = "data"

    def get_train_dataset(self):
        self.train_dataset = GetDataset(
            self.data_dir, 
            self.hparams.train_world, "train")
        return self.train_dataset
    
    def get_test_dataset(self):
        return GetDataset(
            self.data_dir, 
            self.hparams.train_world, "test")