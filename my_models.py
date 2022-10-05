import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms

class NetworkNvidia(nn.Module):
    """NVIDIA model used in the paper."""

    def __init__(self):
        """Initialize NVIDIA model.

        NVIDIA model used
            Image normalization to avoid saturation and make gradients work better.
            Convolution: 5x5, filter: 24, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 36, strides: 2x2, activation: ELU
            Convolution: 5x5, filter: 48, strides: 2x2, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Convolution: 3x3, filter: 64, strides: 1x1, activation: ELU
            Drop out (0.5)
            Fully connected: neurons: 100, activation: ELU
            Fully connected: neurons: 50, activation: ELU
            Fully connected: neurons: 10, activation: ELU
            Fully connected: neurons: 1 (output)

        the convolution layers are meant to handle feature engineering
        the fully connected layer for predicting the steering angle.
        """
        super(NetworkNvidia, self).__init__()
        #self.lambda_layer = LambdaLayer(lambda x: x/255.)
        self.lambda_layer = LambdaLayer(lambda x: x/255. )
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.5)
        )
        self.linear_layers = nn.Sequential(

            nn.Linear(in_features=64 * 13 * 33, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            # nn.Linear(in_features=50, out_features=10),
            # nn.Linear(in_features=10, out_features=1)
        )
        self.fc1 = nn.Linear(in_features=50, out_features=10)
        self.fc2 = nn.Linear(in_features=10, out_features=1)

    def forward(self, input):

        """Forward pass."""
        # input = input.view(input.size(0), 3, 70, 320)
        output = self.lambda_layer(input)
        output = self.conv_layers(output)
        # print(output.shape)
        
        output = output.view(output.size(0), -1)
        
        output = self.linear_layers(output)
        fc1_out = self.fc1(output)
        output = self.fc2(fc1_out)
        return output, fc1_out

class LambdaLayer(nn.Module):

    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class CommaModel(nn.Module):

    def __init__(self):

        super(CommaModel, self).__init__()

        # self.lambda_layer = LambdaLayer(lambda x: x/255.)
        self.lambda_layer = LambdaLayer(lambda x: -1 + 2 * x/255.)
        self.conv_layers = nn.Sequential(
            
            nn.Conv2d(3, 16, 8, stride=4),
            nn.ELU(),

            nn.Conv2d(16, 32, 5, stride=2),
            nn.ELU(),
            
            nn.Conv2d(32, 64, 5, stride=2),
            nn.ELU(),
            nn.Dropout(0.2),
            nn.ELU()
        )

        self.linear_layers = nn.Sequential(
            # hardcoded the in_feature from the output of the prev conv layer
            nn.Linear(in_features=7616, out_features=512),
            nn.Dropout(0.5),
            nn.ELU(),
            nn.Linear(in_features=512, out_features=1)
        )


    def forward(self, input):

        
        input = input.view(input.size(0), 3, 160, 320)
        # lambda_layer = LambdaLayer(lambda x: x/127.5 - 1.)
        output = self.lambda_layer(input)
        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)
        output = self.linear_layers(output)

        return output

if __name__ == "__main__":
    
    # model = NetworkNvidia()
    # lambda_layer = LambdaLayer(lambda x: -1 + 2 * x/255.)
    # lambda_layer = LambdaLayer(lambda x: x/255.)
    # input = torch.tensor([1, 2, 3, 4])
    # print(input)
    # out = lambda_layer(input)
    # print(out)
    # exit()

    model = CommaModel()
    
    print(model)
    input = torch.rand([1, 3, 160, 320])
    out = model(input)
    print(model)