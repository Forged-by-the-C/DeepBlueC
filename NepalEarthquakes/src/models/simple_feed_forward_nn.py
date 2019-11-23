import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import torch 
from src.utils.model_wrapper import model_wrapper


'''

helpful guide

https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb

'''

class Feedforward(torch.nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        self.hidden_size  = hidden_size
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = torch.nn.Linear(self.hidden_size, self.hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc_last = torch.nn.Linear(self.hidden_size, output_size)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        hidden = self.fc1(x)
        relu = self.relu(hidden)
        hidden = self.fc2(hidden)
        relu = self.relu(hidden)
        output = self.fc_last(relu)
        output = self.softmax(output)
        return output

class simple_nn(model_wrapper):

    def init_model(self):
        features = 31
        return Feedforward(features, round(features*0.5), 3)

    def train(self):
        epochs = int(input("Num Epochs?\nInput: "))
        X,y = self.load_data("train")
        x_train = torch.FloatTensor(X)
        #Since y in [1,3] -> [0,2]
        y_train = torch.LongTensor(y-1)
        load = int(input("0: for new model to train \n1: for load model to train\nInput: "))
        if load:
            self.load_model()
            model = self.model
        else:
            model = self.init_model()         
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
        # Compute Loss
        model.train()
        for epoch in range(1, epochs):
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred.squeeze(), y_train)
            labels = np.argmax(y_pred.detach().numpy(), axis=1)
            train_score = f1_score(y_true=y, y_pred=labels, average='micro')
            print('Epoch {}: train loss: {} f1 score: {}'.format(epoch, loss.item(),train_score))
            # Backward pass
            loss.backward()
            optimizer.step()
            if epoch % 1000 == 0:
                self.model = model
                self.save_model()
        save = int(input("Save model: 1 or 0?\nInput: "))
        if save:
            self.model = model
            self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_file_path)

    def load_model(self):
        self.model = self.init_model()
        self.model.load_state_dict(torch.load(self.model_file_path))

    def load_and_score(self):
        self.load_model()
        X,y = self.load_data(split="val")
        x_train = torch.FloatTensor(X)
        #Since y in [1,3] -> [0,2]
        y_train = torch.LongTensor(y-1)
        self.model.eval()
        y_pred = self.model(x_train)
        labels = np.argmax(y_pred.detach().numpy(), axis=1)
        train_score = f1_score(y_true=y, y_pred=labels, average='micro')
        print('Cross val F1 score: {}'.format(train_score))


if __name__ == "__main__":
    mod = simple_nn({"name":"nn1"})
    #mod.train()
    mod.load_and_score()
