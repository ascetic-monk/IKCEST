import torchvision.models as models
from torch.nn import Parameter
import math
import torch
import torch.nn as nn
import numpy as np


class GCN(nn.Module):
    def __init__(self, in_feature, hid_feature, out_feature, use_cuda):
        super(GCN, self).__init__()
        if use_cuda:
            self.gcn1 = GraphConvolution(in_feature, out_feature, bias=True, use_cuda=use_cuda)
            self.gcn2 = GraphConvolution(hid_feature, out_feature, bias=True, use_cuda=use_cuda)
            self.relu1 = nn.LeakyReLU().cuda()
            self.relu2 = nn.LeakyReLU().cuda()
        else:
            self.gcn1 = GraphConvolution(in_feature, hid_feature, bias=True)
            self.gcn2 = GraphConvolution(hid_feature, out_feature, bias=True)
            self.relu1 = nn.LeakyReLU()
            self.relu2 = nn.LeakyReLU()

    def forward(self, input, adj):
        adj = gen_adj(adj)
        x = self.relu1(self.gcn1(input, adj))
        # x = self.relu2(self.gcn2(x, adj))
        return x


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False, use_cuda=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        if use_cuda:
            self.weight = Parameter(torch.Tensor(in_features, out_features).double()).cuda()
            if bias:
                self.bias = Parameter(torch.Tensor(1, 1, out_features).double()).cuda()
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()
        else:
            self.weight = Parameter(torch.Tensor(in_features, out_features).double())
            if bias:
                self.bias = Parameter(torch.Tensor(1, 1, out_features).double())
            else:
                self.register_parameter('bias', None)
            self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def gen_A(num_classes, t, A):
    import pickle
    # result = pickle.load(open(adj_file, 'rb'))
    # _adj = result['adj']
    # _nums = result['nums']
    # _nums = _nums[:, np.newaxis]
    _nums = (A/np.sum(A, axis=1))[:, np.newaxis]
    _adj = A / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0, keepdims=True) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj


def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D).double()
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj
