

# coding simply models 

from turtle import forward
from sklearn.metrics import accuracy_score
import torch.nn as nn 
import numpy as np 
import torch 
from sklearn.datasets import load_iris 
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader 

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size) -> None:
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)
    

    def forward(self,x)-> torch.Tensor:
        x = self.layer1(x)
        x = nn.Sigmoid()(x)
        x = self.layer2(x)
        x = nn.Softmax(dim=1)(x) 
        return x 

def load_dataset(): 
    iris = load_iris()
    X = iris['data']
    y = iris['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0/3, random_state=1)
    print('Shape of training set = ', X_train.shape)
    print('Shape of testing set = ', X_test.shape)
    print('Shape of Training target = ', y_train.shape)
    
    # normalize data
    X_train_norm = (X_train - np.mean(X_train))/np.std(X_train)
    X_train_t = torch.from_numpy(X_train_norm).float()
    y_train_t = torch.from_numpy(y_train)
    
    X_test_norm = (X_test - np.mean(X_train)/np.std(X_train))
    X_test_t = torch.from_numpy(X_test_norm).float()

    y_test_t = torch.from_numpy(y_test)



    return X_train_t, X_test_t, y_train_t, y_test_t 


def train_model(model, data, DEBUG = True ): 
    model.train()
    learning_rate = 0.001 
    if DEBUG:
        num_epochs = 2
    else:
        num_epochs = 100 
    loss_hist = [0]*num_epochs 
    accuracy_hist = [0]*num_epochs 
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

    for epoch in range(num_epochs):
        for x_batch, y_batch in data: 
            
            pred = model(x_batch)
            print(pred, y_batch.reshape(-1,1))
            
            loss = loss_fn(pred, y_batch.reshape(-1,1))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loss_hist[epoch] += loss.item()*y_batch.size(0)
            is_correct = (torch.argmax(pred, dim=1) == y_batch).float()
            accuracy_hist[epoch] += is_correct.mean() 
        
        loss_hist[epoch] /= len(data.dataset)
        accuracy_hist[epoch] /= len(data.dataset)
    print("Epoch = {}, Loss = {:.3f}, Accuracy = {:.3f}".format(epoch, loss_hist[epoch], accuracy_hist[epoch]))
    return model, loss_hist, accuracy_hist 



if __name__ =='__main__':
    X_train, X_test, y_train, y_test  = load_dataset()
    input_size = X_train.shape[1]
    hidden_size = 16
    output_size = 1
    batch_size = 2 
    model = SimpleModel(input_size, hidden_size, output_size)
    print(model)
    y_test_pred = model(X_test)
    print('Model input shape = ', X_test.shape)
    print('Model output shape = ', y_test_pred.shape)
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, )
    #for a in train_dl:
    #    print(a)
    model,loss_hist, accuracy_hist = train_model(model, train_dl)


