import torch
import torch.nn as nn
import torch.nn.functional as F
from models.alexnet import AlexNet
import sys

class SignLoss():
    def __init__(self, kwargs, model, scheme):
        super(SignLoss, self).__init__()
        self.alpha = 0.2  #self.sl_ratio
        self.loss = 0
        self.scheme = scheme 
        self.model = model
        self.kwargs = kwargs 

    def get_loss(self):
        self.reset()
        if isinstance(self.model, AlexNet):
            for m in self.kwargs:
                if self.kwargs[m]['flag'] == True:
                    b = self.kwargs[m]['b']
                    M = self.kwargs[m]['M']
                    
                    b = b.to(torch.device('cuda'))
                    M = M.to(torch.device('cuda'))

                    if self.scheme == 0:    
                        self.loss += (self.alpha * F.relu(-self.model.features[int(m)].scale.view([1, -1]).mm(M).mul(b.view(-1)))).sum()

                    if self.scheme == 1:
                        for i in range(b.shape[0]):
                            if b[i] == -1:
                                b[i] = 0
                        y = self.model.features[int(m)].scale.view([1, -1]).mm(M) 
                        # print(y)
                        loss1 = torch.nn.BCEWithLogitsLoss()
                        self.loss += self.alpha * loss1(y.view(-1), b)

                    if self.scheme == 2:    
                        conv_w = torch.mean(self.model.features[int(m)].conv.weight, dim=0)
                        self.loss += (self.alpha * F.relu(-conv_w.view([1, -1]).mm(M).mul(b.view(-1)))).sum()
                    if self.scheme == 3:
                        for i in range(b.shape[0]):
                            if b[i] == -1:
                                b[i] = 0
                
                        conv_w = torch.mean(self.model.features[int(m)].conv.weight, dim=0)                        
                        y = conv_w.view([1, -1]).mm(M) 
                        
                        # print(y)
                        loss1 = torch.nn.BCEWithLogitsLoss()
                        self.loss += self.alpha * loss1(y.view(-1), b)

        else :
            for sublayer in self.kwargs["layer4"]:
                for module in self.kwargs["layer4"][sublayer]:
                    if self.kwargs["layer4"][sublayer][module]['flag'] == True:
                        b = self.kwargs["layer4"][sublayer][module]['b']
                        M = self.kwargs["layer4"][sublayer][module]['M']

                        b = b.to(torch.device('cuda'))
                        M = M.to(torch.device('cuda'))
                        self.add_resnet_module_loss(sublayer, module, b, M)

        return self.loss

    def reset(self):
        self.loss = 0

    def add_resnet_module_loss(self, sublayer, module, b, M):
        
    
        if module == "convbnrelu_1":
            scale = self.model.layer4[int(sublayer)].convbnrelu_1.scale
            conv_w = torch.mean(self.model.layer4[int(sublayer)].convbnrelu_1.conv.weight, dim = 0) 
        if module == "convbn_2":
            scale = self.model.layer4[int(sublayer)].convbn_2.scale
            conv_w = torch.mean(self.model.layer4[int(sublayer)].convbn_2.conv.weight, dim = 0) 

        if self.scheme == 0:    
            self.loss += (self.alpha * F.relu(-scale.view([1, -1]).mm(M).mul(b.view(-1)))).sum()
        
        if self.scheme == 1:
            for i in range(b.shape[0]):
                if b[i] == -1:
                    b[i] = 0
                    y = scale.view([1, -1]).mm(M) 
                    loss1 = torch.nn.BCEWithLogitsLoss()
                    self.loss += self.alpha * loss1(y.view(-1), b)

        if self.scheme == 2:    
            self.loss += (self.alpha * F.relu(-conv_w.view([1, -1]).mm(M).mul(b.view(-1)))).sum()
                    
        if self.scheme == 3:
            for i in range(b.shape[0]):
                if b[i] == -1:
                    b[i] = 0                      
                    y = conv_w.view([1, -1]).mm(M) 
                    # print(y)
                    loss1 = torch.nn.BCEWithLogitsLoss()
                    self.loss += self.alpha * loss1(y.view(-1), b)
