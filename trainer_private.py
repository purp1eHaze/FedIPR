import time
import copy
from unittest import result
import torch
from torch import tensor
from torch.nn import parameter
import torch.nn.functional as F
import torch.optim as optim 
import numpy as np
from models.losses.sign_loss import SignLoss
from models.alexnet_passport_private import AlexNetPassportPrivate
import time

def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class TesterPrivate(object):
    def __init__(self, model, device, verbose=True):
        self.model = model
        self.device = device
        self.verbose = verbose

    def test_signature(self, kwargs, ind):
        self.model.eval()
        avg_private = 0
        count_private = 0
        
        with torch.no_grad():
            if kwargs != None:
                if isinstance(self.model, AlexNetPassportPrivate):
                    for m in kwargs:
                        if kwargs[m]['flag'] == True:
                            b = kwargs[m]['b']
                            M = kwargs[m]['M']

                            M = M.to(self.device)
                            if ind == 0 or ind == 1:
                                signbit = self.model.features[int(m)].scale.view([1, -1]).mm(M).sign().to(self.device)
                            if ind == 2 or ind == 3:
                                w = torch.mean(self.model.features[int(m)].conv.weight, dim=0)
                                signbit = w.view([1,-1]).mm(M).sign().to(self.device)
                            #print(signbit)

                            privatebit = b
                            privatebit = privatebit.sign().to(self.device)
                    
                            # print(privatebit)
        
                            detection = (signbit == privatebit).float().mean().item()
                            avg_private += detection
                            count_private += 1

                else:
                    for sublayer in kwargs["layer4"]:
                        for module in kwargs["layer4"][sublayer]:
                            if kwargs["layer4"][sublayer][module]['flag'] == True:
                                b = kwargs["layer4"][sublayer][module]['b']
                                M = kwargs["layer4"][sublayer][module]['M']
                                M = M.to(self.device)
                                privatebit = b
                                privatebit = privatebit.sign().to(self.device)

                                if module =='convbnrelu_1':
                                    scale = self.model.layer4[int(sublayer)].convbnrelu_1.scale
                                    conv_w = torch.mean(self.model.layer4[int(sublayer)].convbnrelu_1.conv.weight, dim = 0)
                                if module =='convbn_2':
                                    scale = self.model.layer4[int(sublayer)].convbn_2.scale
                                    conv_w = torch.mean(self.model.layer4[int(sublayer)].convbn_2.conv.weight, dim = 0)
                               
                                if ind == 0 or ind == 1:
                                    signbit = scale.view([1, -1]).mm(M).sign().to(self.device)
                                if ind == 2 or ind == 3:
                                    signbit = conv_w.view([1,-1]).mm(M).sign().to(self.device)
                            #print(signbit)
                            # print(privatebit)
                                detection = (signbit == privatebit).float().mean().item()
                                avg_private += detection
                                count_private += 1

        if kwargs == None:
            avg_private = None
        if count_private != 0:
            avg_private /= count_private

        return avg_private

class TrainerPrivate(object):
    def __init__(self, model, device, dp, sigma):
        self.model = model
        self.device = device
        self.tester = TesterPrivate(model, device)
        self.dp = dp
        self.sigma = sigma

    def _local_update(self, dataloader, wm_dataloader, local_ep, lr, key, ind):
        
        self.optimizer = optim.SGD(self.model.parameters(),
                              lr,
                              momentum=0.9,
                              weight_decay=0.0005) 
                                  
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        train_ldr = dataloader

        for epoch in range(local_ep):
            
            loss_meter = 0
            sign_loss_meter = 0 
            acc_meter = 0 
            
            wm_dataloader = list(wm_dataloader)

            for batch_idx, (x, y) in enumerate(train_ldr):

                x, y = x.to(self.device), y.to(self.device)
                
                wm_batch_idx = int(batch_idx % len(wm_dataloader))  
                (wm_data, wm_target) = wm_dataloader[wm_batch_idx]           
                
                wm_data = wm_data.to(self.device)
                wm_target = wm_target.to(self.device)
                # print(wm_target)

                x = torch.cat([x, wm_data], dim=0)
                y = torch.cat([y, wm_target], dim=0)
                
                self.optimizer.zero_grad()                
  
                if isinstance(key, dict):
                
                    loss = torch.tensor(0.).to(self.device)
                    sign_loss = torch.tensor(0.).to(self.device)

                    pred = self.model(x)
                    loss += F.cross_entropy(pred, y)
                    acc_meter += accuracy(pred, y)[0].item()
                    
                    signloss = SignLoss(key, self.model, ind)
                    sign_loss += signloss.get_loss()
        
                    (loss + sign_loss).backward()
                    self.optimizer.step()
                    sign_loss_meter += sign_loss.item()    

                else:

                    loss = torch.tensor(0.).to(self.device)

                    pred = self.model(x)
                    loss += F.cross_entropy(pred, y)
                    acc_meter += accuracy(pred, y)[0].item()
                    
                    loss.backward()
                    self.optimizer.step()                

                loss_meter += loss.item()        

            loss_meter /= len(train_ldr)
            sign_loss_meter /= len(train_ldr)
            acc_meter /= len(dataloader)

            epoch_loss.append(loss_meter)

        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)


        return self.model.state_dict(), np.mean(epoch_loss), sign_loss_meter


    def _local_update_noback(self, dataloader, wm_dataloader, local_ep, lr, key, ind):
        
        self.optimizer = optim.SGD(self.model.parameters(),
                              lr,
                              momentum=0.9,
                              weight_decay=0.0005) 
                                  
        self.model.to(self.device)
        self.model.train()
        epoch_loss = []
        train_ldr = dataloader

        for epoch in range(local_ep):
            
            loss_meter = 0
            sign_loss_meter = 0 
            acc_meter = 0 
            
            for batch_idx, (x, y) in enumerate(train_ldr):

                x, y = x.to(self.device), y.to(self.device)
                self.optimizer.zero_grad()
                
                if isinstance(key, dict):
                
                    loss = torch.tensor(0.).to(self.device)
                    sign_loss = torch.tensor(0.).to(self.device)

                    pred = self.model(x)
                    loss += F.cross_entropy(pred, y)
                    acc_meter += accuracy(pred, y)[0].item()
                    # sign loss
                    signloss = SignLoss(key, self.model, ind)
                    sign_loss += signloss.get_loss()
                
                    (loss + sign_loss).backward()
                    self.optimizer.step()
                    sign_loss_meter += sign_loss.item()    

                else:

                    loss = torch.tensor(0.).to(self.device)

                    pred = self.model(x)
                    loss += F.cross_entropy(pred, y)
                    acc_meter += accuracy(pred, y)[0].item()

                    loss.backward()
                    self.optimizer.step() 
                
                loss_meter += loss.item()
                   

            loss_meter /= len(train_ldr)
            sign_loss_meter /= len(train_ldr)
            acc_meter /= len(dataloader)

            epoch_loss.append(loss_meter)
            
        if self.dp:
            for param in self.model.parameters():
                param.data = param.data + torch.normal(torch.zeros(param.size()), self.sigma).to(self.device)

        return self.model.state_dict(), np.mean(epoch_loss), sign_loss_meter

    def test(self, dataloader):

        self.model.to(self.device)
        self.model.eval()

        loss_meter = 0
        acc_meter = 0
        runcount = 0

        with torch.no_grad():
            for load in dataloader:
                data, target = load[:2]
                data = data.to(self.device)
                target = target.to(self.device)
        
                pred = self.model(data)  # test = 4
                loss_meter += F.cross_entropy(pred, target, reduction='sum').item() #sum up batch loss
                pred = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
                acc_meter += pred.eq(target.view_as(pred)).sum().item()
                runcount += data.size(0) 

        loss_meter /= runcount
        acc_meter /= runcount

        return  loss_meter, acc_meter

    def fake_test(self, dataloader):

        self.model.to(self.device)
        self.model.eval()

        loss_meter = 0
        acc_meter = 0
        runcount = 0

        with torch.no_grad():
            for load in dataloader:
                data, target = load[:2]
                data = data.to(self.device)
                target = target.to(self.device)
        
                pred = self.model(data)  # test = 4
                #loss_meter += F.cross_entropy(pred, target, reduction='sum').item() #sum up batch loss
                pred_result = pred.max(1, keepdim=True)[1] # get the index of the max log-probability
                fake_result = pred_result.view_as(target)
               
                loss_meter += F.cross_entropy(pred, fake_result, reduction='sum').item()
                acc_meter += pred_result.eq(target.view_as(pred_result)).sum().item()
                runcount += data.size(0) 

        loss_meter /= runcount
        acc_meter /= runcount

        return  loss_meter, acc_meter

 
