import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
from torch.nn.parameter import Parameter

class LinearCapsPro(nn.Module):
    def __init__(self, in_features, num_C, num_D, eps=0.0001):
        super(LinearCapsPro, self).__init__()
        self.in_features = in_features
        self.num_C = num_C
        self.num_D = num_D
        self.eps = eps
        self.weight = Parameter(torch.Tensor(num_C, num_D, self.in_features))
        self.eye = Parameter(torch.eye(num_D), requires_grad=False)
        self.count = 0
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        weight_caps = torch.matmul(self.weight, self.weight.permute(0, 2, 1))
        sigma = torch.inverse(weight_caps + self.eps * self.eye)

        out = torch.matmul(x, torch.t(self.weight.view(-1, x.size(-1))))
        out = out.view(out.shape[0], self.num_C, 1, self.num_D)
        out = torch.matmul(out, sigma)
        out = torch.matmul(out, self.weight)
        out = torch.squeeze(out, dim=2)
        out = torch.matmul(out, x.unsqueeze(dim=2)).squeeze(dim=2)
        out = torch.sqrt(out)
        return out

class MCCP(nn.Module):
    def __init__(self, in_features, num_C, num_D, reciptive=512, strides=256, eps=0.0001):
        super(MCCP, self).__init__()
        self.in_features = in_features
        print('the dimension of input vector is:', in_features)
        print('mccp reciptive is:', reciptive)
        print('mccp strides is:', strides)
        self.reciptive = reciptive
        self.strides = strides
        self.num_C = num_C
        self.num_D = num_D
        self.eps = eps
        self.eye = Parameter(torch.eye(num_D), requires_grad=False)
        self.weight = Parameter(torch.Tensor(num_C, num_D, self.reciptive))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(2))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        weight_caps = torch.matmul(self.weight, self.weight.permute(0, 2, 1))
        sigma = torch.inverse(weight_caps + self.eps * self.eye)
        # implemented as a convolutional procedure
        # vec2mat = nn.functional.unfold(torch.cat((x, x[:, :self.reciptive]), dim=-1).unsqueeze(1).unsqueeze(1), (1, self.reciptive), stride=self.strides)
        # matrix = vec2mat.permute(0, 2, 1)
        # b, n, d = matrix.size()
        # out1 = torch.matmul(matrix, torch.t(self.weight.view(-1, d)))
        # out1 = out1.view(b, n, self.num_C, 1, self.num_D)
        # out1 = torch.matmul(out1, sigma)  # (b, n, num_C, 1, num_D)
        # out2 = torch.matmul(self.weight.unsqueeze(dim=0).unsqueeze(dim=0), matrix.unsqueeze(dim=2).unsqueeze(dim=-1))
        # out2 = torch.matmul(out1, out2) # (b, n, num_C, 1, 1)
        # return out2.sum(dim=1).sqrt().squeeze()

        # implemented column by column
        results = []
        for i in range(0, x.shape[1], self.strides):
            # vec2mat
            if i + self.reciptive > x.shape[1]:
                inputs = torch.cat((x[:, i:], x[:, :(self.reciptive-x.shape[1]+i)]), dim=-1)
            else:
                inputs = x[:, i:i+self.reciptive]
            # project
            out = torch.matmul(inputs, torch.t(self.weight.view(-1, inputs.size(-1))))
            out = out.view(out.shape[0], self.num_C, 1, self.num_D)
            out = torch.matmul(out, sigma)
            out = torch.matmul(out, self.weight.view(self.num_C, self.num_D, self.reciptive))
            out = torch.squeeze(out, dim=2)
            out = torch.matmul(out, torch.unsqueeze(inputs, dim=2))
            results.append(torch.sqrt(out))
        results = torch.cat(results, dim=-1)
        return torch.mean(results, dim=-1)
