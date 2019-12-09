import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier
import time
import torch 
from src.utils.model_wrapper import model_wrapper


'''

helpful guide

https://medium.com/biaslyai/pytorch-introduction-to-neural-network-feedforward-neural-network-model-e7231cff47cb

'''

class Feedforward(torch.nn.Module):

    def __init__(self, input_size, output_size):
        super(Feedforward, self).__init__()
        self.input_size = input_size
        n_layers = 4
        hidden_size_list  = [round(input_size *(0.5 ** i)) for i in range(1,n_layers)]
        self.fc0 = torch.nn.Linear(self.input_size, hidden_size_list[0])
        self.fc1 = torch.nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc2 = torch.nn.Linear(hidden_size_list[1], hidden_size_list[2])
        self.fc_last = torch.nn.Linear(hidden_size_list[-1], output_size)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        hidden = self.relu(self.fc0(x))
        hidden = self.relu(self.fc1(hidden))
        hidden = self.relu(self.fc2(hidden))
        output = self.relu(self.fc_last(hidden))
        output = self.softmax(output)
        return output

class simple_nn(model_wrapper):

    def init_model(self, input_size, output_size=3, load=0):
        self.model = Feedforward(input_size, output_size)
        if load:
            self.model.load_state_dict(torch.load(self.model_file_path))

    def train(self):
        epochs = int(input("Num Epochs?\nInput: "))
        X,y = self.load_data("train")
        x_train = torch.FloatTensor(X)
        #Since y in [1,3] -> [0,2]
        y_train = torch.LongTensor(y-1)
        features = X.shape[1]         
        load = int(input("0: for new model to train \n1: for load model to train\nInput: "))
        self.init_model(features, 3, load)
        criterion = torch.nn.CrossEntropyLoss()
        #TODO: Learning Rate Scheduler?
        optimizer = torch.optim.Adadelta(self.model.parameters())
        # Put model in training mode
        train_start = time.time()
        self.model.train()
        for epoch in range(1, epochs):
            optimizer.zero_grad()
            y_pred = self.model(x_train)
            loss = criterion(y_pred.squeeze(), y_train)
            labels = np.argmax(y_pred.detach().numpy(), axis=1)
            train_score = f1_score(y_true=y_train, y_pred=labels, average='micro')
            print('{} :: Epoch {}: train loss: {} f1 score: {}'.format(
                time.strftime("%H:%M:%S", time.gmtime(time.time() - train_start)), 
                epoch, loss.item(),train_score))
            # Backward pass
            loss.backward()
            optimizer.step()
            if epoch % 20 == 0:
                #Dynamically update loss weights based on Confusion matrix
                conf_matrix = self.gen_conf_matrix(y_train, labels)
                error = np.sum(conf_matrix*(1 - np.identity(conf_matrix.shape[0])), axis=1)
                print("Epoch: {} Error: {}".format(epoch, error))
                weight = torch.from_numpy(error).float()
                criterion = torch.nn.CrossEntropyLoss(weight=weight)
            if epoch % 1000 == 0:
                self.save_model()
        save = int(input("Save model: 1 or 0?\nInput: "))
        if save:
            self.save_model()

    def save_model(self):
        torch.save(self.model.state_dict(), self.model_file_path)

    def load_and_score(self):
        X,y = self.load_data(split="val")
        x_train = torch.FloatTensor(X)
        #Since y in [1,3] -> [0,2]
        y_train = torch.LongTensor(y-1)
        features = X.shape[1]         
        self.init_model(features, 3, load=True)
        self.model.eval()
        y_pred = self.model(x_train)
        labels = np.argmax(y_pred.detach().numpy(), axis=1)
        train_score = f1_score(y_true=y_train, y_pred=labels, average='micro')
        print('Cross val F1 score: {}'.format(train_score))
        conf_matrix = self.gen_conf_matrix(y_train, labels)
        print('Confusion Matrix: \n', conf_matrix)
        self.plot_conf_matrix(conf_matrix)
        #print(np.sum(conf_matrix*(1 - np.identity(conf_matrix.shape[0])), axis=1))

if __name__ == "__main__":
    mod = simple_nn({"name":"nn1_weighted"})
    #mod.train()
    mod.load_and_score()
