import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet101

import dgl
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv, GATConv
from torchvision.models.resnet import resnet152

class GCNLabelEncoder(nn.Module):
    def __init__(self, graph: DGLGraph, emb_dim=512, use_bias=True, num_hops=2, dropout=0.2):
        super().__init__()
        self.graph = graph
        self.emb_dim = emb_dim
        self.num_nodes = graph.num_nodes()
        self.use_bias = use_bias
        self.num_hops = num_hops
        self.dropout = dropout
        self.init_emb = nn.Parameter(torch.Tensor(self.num_nodes, 300))
        nn.init.kaiming_normal_(self.init_emb, a=math.sqrt(5))
        if num_hops == 1:
            self.gcn1 = GraphConv(300, emb_dim, weight=True, bias=use_bias)
        elif num_hops == 2:
            self.gcn1 = GraphConv(300, 400, weight=True, bias=use_bias)
            self.act1 = nn.LeakyReLU(0.2)
            self.drop1 = nn.Dropout(dropout)
            self.gcn2 = GraphConv(400, emb_dim, weight=True, bias=use_bias)
        elif num_hops == 3:
            self.gcn1 = GraphConv(300, 400, weight=True, bias=use_bias)
            self.act1 = nn.LeakyReLU(0.2)
            self.drop1 = nn.Dropout(dropout)
            self.gcn2 = GraphConv(400, emb_dim, weight=True, bias=use_bias)
            self.act2 = nn.LeakyReLU(0.2)
            self.drop2 = nn.Dropout(dropout)
            self.gcn3 = GraphConv(emb_dim, emb_dim, weight=True, bias=use_bias)
    def forward(self):
        if self.num_hops == 1:
            x = self.gcn1(self.graph, self.init_emb)
        elif self.num_hops == 2:
            x = self.gcn1(self.graph, self.init_emb)
            x = self.act1(x)
            x = self.drop1(x)
            x = self.gcn2(self.graph, x)
        elif self.num_hops == 3:
            x = self.gcn1(self.graph, self.init_emb)
            x = self.act1(x)
            x = self.drop1(x)
            x = self.gcn2(self.graph, x)
            x = self.act2(x)
            x = self.drop2(x)
            x = self.gcn3(self.graph, x)
        return x                             # [num_nodes, emb_dim]

class GCNLabelEncoder2048(nn.Module):
    def __init__(self, graph: DGLGraph, emb_dim=2048, use_bias=True, num_hops=2, dropout=0.):
        super().__init__()
        self.graph = graph
        self.emb_dim = emb_dim
        self.num_nodes = graph.num_nodes()
        self.use_bias = use_bias
        self.num_hops = num_hops
        self.dropout = dropout
        self.init_emb = nn.Parameter(torch.Tensor(self.num_nodes, 1024))
        nn.init.kaiming_normal_(self.init_emb, a=math.sqrt(5))
        if num_hops == 1:
            self.gcn1 = GraphConv(1024, emb_dim, weight=True, bias=use_bias)
        elif num_hops == 2:
            self.gcn1 = GraphConv(1024, 1024, weight=True, bias=use_bias)
            self.act1 = nn.LeakyReLU(0.2)
            self.drop1 = nn.Dropout(dropout)
            self.gcn2 = GraphConv(1024, emb_dim, weight=True, bias=use_bias)
        elif num_hops == 3:
            self.gcn1 = GraphConv(1024, 1024, weight=True, bias=use_bias)
            self.act1 = nn.LeakyReLU(0.2)
            self.drop1 = nn.Dropout(dropout)
            self.gcn2 = GraphConv(1024, emb_dim, weight=True, bias=use_bias)
            self.act2 = nn.LeakyReLU(0.2)
            self.drop2 = nn.Dropout(dropout)
            self.gcn3 = GraphConv(emb_dim, emb_dim, weight=True, bias=use_bias)
    def forward(self):
        if self.num_hops == 1:
            x = self.gcn1(self.graph, self.init_emb)
        elif self.num_hops == 2:
            x = self.gcn1(self.graph, self.init_emb)
            x = self.act1(x)
            x = self.drop1(x)
            x = self.gcn2(self.graph, x)
        elif self.num_hops == 3:
            x = self.gcn1(self.graph, self.init_emb)
            x = self.act1(x)
            x = self.drop1(x)
            x = self.gcn2(self.graph, x)
            x = self.act2(x)
            x = self.drop2(x)
            x = self.gcn3(self.graph, x)
        return x                             # [num_nodes, emb_dim]

def init_layer(layer, nonlinearity='leaky_relu'):
    """Initialize a Linear or Convolutional layer. """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)
    
    
def init_bn(bn):
    """Initialize a Batchnorm layer. """
    
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        
        super(ConvBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.conv2 = nn.Conv2d(in_channels=out_channels, 
                              out_channels=out_channels,
                              kernel_size=(3, 3), stride=(1, 1),
                              padding=(1, 1), bias=False)
                              
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.init_weights()
        
    def init_weights(self):
        
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        
    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        else:
            raise Exception('Incorrect argument!')
        
        return x
    
    
class Cnn_9layers_AvgPooling(nn.Module):
    
    def __init__(self, classes_num):
        super(Cnn_9layers_AvgPooling, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc_noisy = nn.Linear(512, classes_num, bias=True)
        self.fc_curated = nn.Linear(512, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_noisy)
        init_layer(self.fc_curated)

    def forward(self, input, is_curated):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        is_curated = is_curated.view(-1, 1)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        logits_noisy = self.fc_noisy(x)
        logits_curated = self.fc_curated(x)
        logits = is_curated * logits_curated + (1-is_curated) * logits_noisy
        output = torch.sigmoid(logits)
        
        return output

class ResNet_V101_AvgPooling(nn.Module):
    
    def __init__(self, classes_num):
        super(ResNet_V101_AvgPooling, self).__init__()

        resnet = resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)

        self.fc_noisy = nn.Linear(2048, classes_num, bias=True)
        self.fc_curated = nn.Linear(2048, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_noisy)
        init_layer(self.fc_curated)

    def forward(self, input, is_curated):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        x = torch.stack([input,input,input], dim=1)  # 3 channels
        # x = input[:, None, :, :]
        '''(batch_size, 3, times_steps, freq_bins)'''
        is_curated = is_curated.view(-1, 1)
        x = self.feature_extractor(x).view(-1, 2048)
        logits_noisy = self.fc_noisy(x)
        logits_curated = self.fc_curated(x)
        logits = is_curated * logits_curated + (1-is_curated) * logits_noisy
        output = torch.sigmoid(logits)
        
        return output


class Cnn_9layers_AvgPooling2(nn.Module):
    
    def __init__(self, classes_num, emb_dim=512):
        super(Cnn_9layers_AvgPooling2, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.fc = nn.Linear(512, emb_dim)
        self.fc_noisy = nn.Linear(emb_dim, classes_num, bias=True)
        self.fc_curated = nn.Linear(emb_dim, classes_num, bias=True)

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_noisy)
        init_layer(self.fc_curated)

    def forward(self, input, is_curated):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        is_curated = is_curated.view(-1, 1)
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        x = self.fc(x)
        logits_noisy = self.fc_noisy(x)
        logits_curated = self.fc_curated(x)
        logits = is_curated * logits_curated + (1-is_curated) * logits_noisy
        output = torch.sigmoid(logits)
        
        return output


class Cnn_9layers_AvgPooling_Emb(nn.Module):
    def __init__(self, classes_num, emb_dim=512):
        
        super(Cnn_9layers_AvgPooling_Emb, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.emb_dim = emb_dim
        self.fc_noisy = nn.Linear(512, emb_dim, bias=True)
        self.fc_curated = nn.Linear(512, emb_dim, bias=True)

        # self.fc = nn.Linear(512, classes_num, bias=True)
        self.label_emb = nn.Parameter(torch.Tensor(emb_dim, classes_num))
        
        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_noisy)
        init_layer(self.fc_curated)
        nn.init.kaiming_normal_(self.label_emb, a=math.sqrt(5))

    def forward(self, input, is_curated):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        is_curated = is_curated.view(-1, 1)
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        logits_noisy = self.fc_noisy(x).mm(self.label_emb)
        logits_curated = self.fc_curated(x).mm(self.label_emb)
        logits = is_curated * logits_curated + (1-is_curated) * logits_noisy
        output = torch.sigmoid(logits)
        
        return output

class Cnn_9layers_AvgPooling_GCNEmb(nn.Module):
    def __init__(self, classes_num, graph, class_indices, emb_dim=512):
        
        super(Cnn_9layers_AvgPooling_GCNEmb, self).__init__()

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)

        self.emb_dim = emb_dim
        self.fc_noisy = nn.Linear(512, emb_dim, bias=True)
        self.fc_curated = nn.Linear(512, emb_dim, bias=True)

        self.graph_encoder = GCNLabelEncoder(graph, emb_dim)
        self.class_indices = class_indices
        # workaround to easily get a model's device
        self.dummy_param = nn.Parameter(torch.empty(0))

        # self.init_weights()

    def init_weights(self):
        init_layer(self.fc_noisy)
        init_layer(self.fc_curated)

    def forward(self, input, is_curated):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        is_curated = is_curated.view(-1, 1)
        x = input[:, None, :, :]
        '''(batch_size, 1, times_steps, freq_bins)'''
        
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = self.conv_block4(x, pool_size=(1, 1), pool_type='avg')
        '''(batch_size, feature_maps, time_steps, freq_bins)'''
        
        x = torch.mean(x, dim=3)        # (batch_size, feature_maps, time_stpes)
        (x, _) = torch.max(x, dim=2)    # (batch_size, feature_maps)
        label_emb = self.graph_encoder()   # (num_nodes, feature_maps)
        label_emb = label_emb[self.class_indices]   # (num_classes, feature_maps)
        label_emb = label_emb.T
        logits_noisy = self.fc_noisy(x).mm(label_emb)
        logits_curated = self.fc_curated(x).mm(label_emb)
        logits = is_curated * logits_curated + (1-is_curated) * logits_noisy
        output = torch.sigmoid(logits)
        
        return output

class ResNet_V101_AvgPooling_GCNEmb(nn.Module):
    
    def __init__(self, classes_num, graph, class_indices, emb_dim=2048):
        super(ResNet_V101_AvgPooling_GCNEmb, self).__init__()

        resnet = resnet101(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.feature_extractor = nn.Sequential(*modules)

        self.fc_noisy = nn.Linear(2048, 2048, bias=True)
        self.fc_curated = nn.Linear(2048, 2048, bias=True)
        self.graph_encoder = GCNLabelEncoder2048(graph, emb_dim)
        self.class_indices = class_indices
        # workaround to easily get a model's device
        self.dummy_param = nn.Parameter(torch.empty(0))

        self.init_weights()

    def init_weights(self):
        init_layer(self.fc_noisy)
        init_layer(self.fc_curated)

    def forward(self, input, is_curated):
        '''
        Input: (batch_size, times_steps, freq_bins)'''
        x = torch.stack([input,input,input], dim=1)  # 3 channels
        # x = input[:, None, :, :]
        '''(batch_size, 3, times_steps, freq_bins)'''
        is_curated = is_curated.view(-1, 1)
        x = self.feature_extractor(x).view(-1, 2048)
        label_emb = self.graph_encoder()   # (num_nodes, feature_maps)
        label_emb = label_emb[self.class_indices]   # (num_classes, feature_maps)
        label_emb = label_emb.T
        logits_noisy = self.fc_noisy(x).mm(label_emb)
        logits_curated = self.fc_curated(x).mm(label_emb)
        logits = is_curated * logits_curated + (1-is_curated) * logits_noisy
        output = torch.sigmoid(logits)
        
        return output
