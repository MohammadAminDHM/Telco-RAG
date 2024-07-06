import torch
import torch.nn as nn
import torch.nn.functional as F

class NNRouter(nn.Module):
    def __init__(self):
        super(NNRouter, self).__init__()
        self.layer1_1 = nn.Linear(1024, 768)
        self.layer1_2 = nn.Linear(768, 512)
        self.layer1_3 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.2)  

        self.layer2_1 = nn.Linear(18, 128)
        self.layer2_2 = nn.Linear(128, 256)
        self.dropout2 = nn.Dropout(0.05) 
        
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.alfa = nn.Parameter(torch.ones(1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)

        self.output_layer1 = nn.Linear(256, 128)
        self.output_layer2 = nn.Linear(128, 18)
        
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, input_1, input_2):

        x1 = F.relu(self.layer1_1(input_1))
        x1 = self.dropout1(x1)
        x1 = F.relu(self.layer1_2(x1))
        x1 = self.dropout1(x1)
        x1 = F.relu(self.layer1_3(x1))
        x1 = self.batchnorm1(x1)

        x2 = F.relu(self.layer2_1(input_2))
        x2 = self.dropout2(x2)
        x2 = F.relu(self.layer2_2(x2))
        x2 = self.batchnorm2(x2)

        weighted_x1 = self.alfa * x1
        weighted_x2 = self.beta * x2
        
        combined = weighted_x1 + weighted_x2
        
        output = self.output_layer1(self.leaky_relu(combined))
        output = self.output_layer2(self.leaky_relu(output))
        
        return output
    