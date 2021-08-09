from torch.utils.data import Dataset, DataLoader

class Custom_Training_Dataset(Dataset):
    def __init__(self,X_train,y_train):
        self.X_train = X_train.to_numpy()
        self.y_train = y_train.to_numpy()
    def __len__(self):
        return len(self.X_train)
    def __getitem__(self,index):
        X = self.X_train[index]
        y = self.y_train[index]
        return X,y
    
class Custom_Validation_Dataset(Dataset):
    def __init__(self,X_valid,y_valid):
        self.X_valid = X_valid.to_numpy()
        self.y_valid = y_valid.to_numpy()
    def __len__(self):
        return len(self.X_valid)
    def __getitem__(self,index):
        X = self.X_valid[index]
        y = self.y_valid[index]
        return X,y
    
class Custom_Testing_Dataset(Dataset):
    def __init__(self,X_test,y_test):
        self.X_test = X_test.to_numpy()
        self.y_test = y_test.to_numpy()
    def __len__(self):
        return len(self.X_test)
    def __getitem__(self,index):
        X = self.X_test[index]
        y = self.y_test[index]
        return X,y