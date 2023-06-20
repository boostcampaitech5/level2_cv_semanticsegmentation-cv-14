import torch
import torch.nn as nn
import torch.nn.functional as F
def binary_cross_entropy(inputs, targets):
    loss = torch.mean(-targets * torch.log(inputs) - (1 - targets) * torch.log(1 - inputs))
    return loss
class Focal_loss(nn.Module):
    def __init__(self):
        super(Focal_loss, self).__init__()
        
    def __call__(self, inputs, targets):
        return self.Floss(inputs, targets)
    
    def Floss(self,inputs, targets, alpha=.25, gamma=2):
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # custom BCE function
        # BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE = torch.mean(-targets * torch.log(inputs) - (1 - targets) * torch.log(1 - inputs))
        BCE_EXP = torch.exp(-BCE)
        loss = alpha * (1-BCE_EXP)**gamma * BCE
        return loss

class IOU_loss(nn.Module):
    def __init__(self):
        super(IOU_loss, self).__init__()

    def __call__(self, inputs, targets, smooth=1):
        return self.Iloss(inputs, targets, smooth)

    def Iloss(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection
        IoU = (intersection+smooth) / (union + smooth)
        return 1 - IoU
    
class Dice_loss(nn.Module):
    def __init__(self):
        super(Dice_loss, self).__init__()   

    def __call__(self, inputs, targets, smooth=1e-3):
        return self.Dloss(inputs, targets, smooth)

    def Dloss(self, pred, target, smooth=1e-3):
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = 1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth))
        return loss.mean()
    
class Calc_loss(nn.Module):
    def __init__(self):
        super(Calc_loss, self).__init__()

    def __call__(self, inputs, targets, bce_weight=0.5, smooth=1.):
        return self.calc_loss(inputs, targets, bce_weight, smooth)

    def calc_loss(self, pred, target, bce_weight=0.5, smooth=1.):

        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred = torch.sigmoid(pred)
        
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        dice = loss.mean()
        loss = bce * bce_weight + dice * (1 - bce_weight)
        return loss



class Dice_Focal_loss(nn.Module):
    def __init__(self):
        super(Dice_Focal_loss, self).__init__()

    def __call__(self, inputs, targets, bce_weight=0.5, smooth=1.):
        return self.calc_loss(inputs, targets)#, bce_weight, smooth)

    def calc_loss(self, pred, target, bce_weight=0.5, smooth=1.,alpha=.25, gamma=2):
        pred = torch.sigmoid(pred)
        # Focal loss
        inputs = pred.view(-1)
        outputs = target.view(-1)
        BCE = F.binary_cross_entropy(inputs, outputs, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        Floss = alpha * (1-BCE_EXP)**gamma * BCE
        
        # dice loss
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred = pred.contiguous()
        target = target.contiguous()
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        Dloss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) + target.sum(dim=2).sum(dim=2) + smooth)))
        dice = Dloss.mean()
        Dloss = bce * bce_weight + dice * (1 - bce_weight)
        return Dloss+Floss

# loss_fn = IOU_loss()
# x = torch.randn(8,29,512,512)
# y = torch.randn(8,29,512,512)
# loss = loss_fn(x, y)
# print(loss)

"""
import torch
import torch.nn.functional as F

class Lossfunction():
    
    # focal loss 
    def focal_loss(inputs, targets, alpha=.25, gamma=2) : 
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        loss = alpha * (1-BCE_EXP)**gamma * BCE
        return loss 
    
    
    def IOU_loss(inputs, targets, smooth=1) : 
        inputs = F.sigmoid(inputs)      
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection 
        IoU = (intersection + smooth)/(union + smooth)
        return 1 - IoU
    
    
    def dice_loss(pred, target, smooth = 1.):
        pred = pred.contiguous()
        target = target.contiguous()   
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
        return loss.mean()


    def calc_loss(pred, target, bce_weight = 0.5,smooth = 1.):
        pred = pred.contiguous()
        target = target.contiguous()  
        intersection = (pred * target).sum(dim=2).sum(dim=2)
        loss = (1 - ((2. * intersection + smooth) / (pred.sum(dim=2).sum(dim=2) +   target.sum(dim=2).sum(dim=2) + smooth)))
        
        bce = F.binary_cross_entropy_with_logits(pred, target)
        pred = F.sigmoid(pred)
        dice = loss.mean()
        loss = bce * bce_weight + dice * (1 - bce_weight)
        return loss
"""