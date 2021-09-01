import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import DataSet

SPECIES_OF_DATA = 6 # relative date, temp, wind velocity, road_type, snowfall, rainfall
LENGTH_OF_DATE = 7 # 1week
EPOCH = 100
LEARNING_RATE = 0.01
THRESHOLD = 0.5

# def train(epoch, model, loader_train, optimizer):
#     model.train()

#     for data, targets in loader_train:
#         optimizer.zero_grad()
#         outputs = model(data)
#         loss = nn.CrossEntropyLoss(outputs, targets)

#         loss.backward()
#         optimizer.step()
    
#     print(f"epoch: {epoch}")


def test(model, loader_test):
    model.eval()
    correct = 0

    with torch.no_grad():
        for data, target in loader_test:
            outputs = model(data)
            #print(f"test-output = {outputs.data}")
            #print(f"target = {target.data}")
            #print(f"data = {data.data}")
            for index in range(len(data)):
                if outputs[index] > THRESHOLD: # predict there is crash
                    if target[index][0] == 1: # crash
                        correct += 1 
                else: # predect there is no crash
                    if target[index] == 0: # no crash
                        correct += 1
    
    data_num = len(loader_test.dataset)
    print(f"accuracy: {float(correct) / float(data_num) * 100}")

    
# learning dataset
# [relative date, temp, wind velocity, road_type, snowfall, rainfall, relative date ...] ...

if __name__ == "__main__":
    input_layer = SPECIES_OF_DATA * LENGTH_OF_DATE
    hidden_layer1 = int(input_layer / 2.0)
    hidden_layer2 = int(hidden_layer1 / 2.0)
    output_layer = 1

    model = nn.Sequential()
    model.add_module('fc1', nn.Linear(input_layer, hidden_layer1))
    model.add_module('relu1', nn.ReLU())
    model.add_module('fc2', nn.Linear(hidden_layer1, hidden_layer2))
    model.add_module('relu2', nn.ReLU())
    model.add_module('fc3', nn.Linear(hidden_layer2, output_layer))

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    total_x = DataSet.total_x
    total_y = DataSet.total_y

    learning_dataset, test_dataset_x, crash_dataset, test_dataset_y = train_test_split(total_x, total_y, test_size=1/10, random_state=0)

    learning_dataset = torch.FloatTensor(learning_dataset)
    test_dataset_x = torch.FloatTensor(test_dataset_x)
    crash_dataset = torch.LongTensor(crash_dataset)
    test_dataset_y = torch.LongTensor(test_dataset_y)

    dataset_train = TensorDataset(learning_dataset, crash_dataset)
    dataset_test = TensorDataset(test_dataset_x, test_dataset_y)

    loader_train = DataLoader(dataset_train, batch_size=8, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=8, shuffle=False)

    for data, target in loader_train:
        print(f"loader_train = {data}, {target}")

    for data, target in loader_test:
        print(f"loader_test = {data}, {target}")

    test(model, loader_test)

    model.train()
    for epoch in range(EPOCH):
        for data, targets in loader_train:
            print(f"data = {data}, len = {len(data)}")
            print(f"target = {targets}, len = {len(targets)}")
            
            outputs = model(data)
            print(f"output = {outputs.data}, len = {len(outputs)}")
            print(f"shape: output = {outputs.shape}, targets = {targets.shape}")
            loss = loss_function(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch: {epoch} | loss = {loss.item()}")

    print("==== after learning ====")
    test(model, loader_test)