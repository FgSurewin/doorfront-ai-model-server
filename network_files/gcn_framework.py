from torch.nn import Parameter
from network_files.gcn_util import *
import pickle
import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
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


class GCNnet(nn.Module):
    def __init__(self, num_classes, adj_file=None, in_channel=300, t=0):
        super(GCNnet, self).__init__()
        self.num_classes = 4
        self.t = t
        self.adj_file = adj_file
        self.A = 0
        self.gc1 = GraphConvolution(in_channel, 1024)
        # self.gc2 = GraphConvolution(512, 1024)
        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=1)

        # image normalization
        self.image_normalization_mean = [0.485, 0.456, 0.406]
        self.image_normalization_std = [0.229, 0.224, 0.225]

    def forward(self, feature, inp):
        _adj = gen_A(self.num_classes, self.t, self.adj_file)
        self.A = Parameter(torch.from_numpy(_adj).float())
        adj = gen_adj(self.A).detach()
        tensor_inp = Parameter(inp)
        x = self.gc1(tensor_inp, adj)
        x = self.relu(x)
        # x = self.gc2(x, adj)

        x = x.transpose(0, 1)
        x = torch.matmul(feature, x)
        # x = self.softmax(x)
        return x

    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.gc1.parameters(), 'lr': lr},
                {'params': self.gc2.parameters(), 'lr': lr},
                ]


def gcn_net(pro_features, num_classes, t, adj_file=None, in_channel=300):
    return GCNnet(pro_features, num_classes, t=t, adj_file=adj_file, in_channel=in_channel)
