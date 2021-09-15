import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import DataSet

SPECIES_OF_DATA = 6 # relative date, temp, wind velocity, road_type, snowfall, rainfall
LENGTH_OF_DATE = 7 # 1week
EPOCH = 100
LEARNING_RATE = 0.01
THRESHOLD = 0.5
MINIBATCH = 8


# learning dataset
# [relative date, temp, wind velocity, road_type, snowfall, rainfall, relative date ...] ...

if __name__ == "__main__":
    input_layer = SPECIES_OF_DATA * LENGTH_OF_DATE
    hidden_layer1_front = int(input_layer / 2.0)
    hidden_layer1_rear = hidden_layer2_front = hidden_layer2_rear = int(hidden_layer1_front / 2.0)
    output_layer = 1

    model = nn.Sequential()
    model.add_module('fc1', nn.Linear(input_layer, hidden_layer1_front))
    model.add_module('relu1', nn.ReLU())
    model.add_module('fc2', nn.Linear(hidden_layer1_front, hidden_layer1_rear))
    model.add_module('relu2', nn.ReLU())
    model.add_module('fc3', nn.Linear(hidden_layer2_front, hidden_layer2_rear))
    model.add_module("softmax1", nn.Softmax(dim=1))
    model.add_module("fc4", nn.Linear(hidden_layer2_rear, output_layer))

    loss_function = torch.nn.BCELoss()
   

    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    total_x = DataSet.total_x
    normalizated_x = F.softmax(torch.FloatTensor(total_x), dim=1)
    total_y = DataSet.total_y
    print(f"dataset:\nx:{list(normalizated_x.numpy())}\ny:{total_y}")

    learning_dataset, test_dataset_x, crash_dataset, test_dataset_y = train_test_split(normalizated_x, total_y, test_size=1/10, random_state=0)

    learning_dataset = torch.FloatTensor(learning_dataset)
    test_dataset_x = torch.FloatTensor(test_dataset_x)
    crash_dataset = torch.FloatTensor(crash_dataset)
    test_dataset_y = torch.FloatTensor(test_dataset_y)

    dataset_train = TensorDataset(learning_dataset, crash_dataset)
    dataset_test = TensorDataset(test_dataset_x, test_dataset_y)

    loader_train = DataLoader(dataset_train, batch_size=MINIBATCH, shuffle=True)
    loader_test = DataLoader(dataset_test, batch_size=MINIBATCH, shuffle=False)

    model.train() # set train mode
    for epoch in range(EPOCH):
        for data, targets in loader_train:
            outputs = model(data)
            normalizated_output = F.softmax(torch.FloatTensor(outputs), dim=1)
            loss = loss_function(normalizated_output, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch: {epoch} | loss = {round(loss.item()*100.0, 2)} %")

    print("==== after learning ====")
    print("----- TEST -----")
    model.eval() # set test mode
    correct = 0

    with torch.no_grad():
        for data, target in loader_test:
            outputs = model(data)
            for index in range(len(data)):
                if outputs[index] > THRESHOLD: # predict there is crash
                    if target[index][0] == 1: # crash
                        correct += 1 
                else: # predect there is no crash
                    if target[index] == 0: # no crash
                        correct += 1
    
    data_num = len(loader_test.dataset)
    print(f"accuracy: {float(correct) / float(data_num) * 100} %")