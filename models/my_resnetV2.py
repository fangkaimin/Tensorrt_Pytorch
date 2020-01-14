import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual_block(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        self.stride = stride
        super(Residual_block,self).__init__()
        self.Conv_01 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.Conv_02 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.Bn_01 = nn.BatchNorm2d(planes)
        self.Bn_02 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.Bn_01(self.Conv_01(x)))
        out = self.Bn_02(self.Conv_02(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):    
    def __init__(self,block,num_blocks,num_classes=100):
        super(ResNet,self).__init__()
        cfg = [(3),(32),(64),(128),(256)]
        self.in_planes = cfg[1]
        
        self.Conv_00 = nn.Conv2d(cfg[0],cfg[1],kernel_size=3, stride=1,padding=1,bias=False)
        self.Bn_00 = nn.BatchNorm2d(cfg[1])

        self.Conv_01 = self._Make_Layer(block, cfg[1], num_blocks[0], stride=1)
        self.Conv_02 = self._Make_Layer(block, cfg[2], num_blocks[1], stride=2)
        self.Conv_03 = self._Make_Layer(block, cfg[3], num_blocks[2], stride=2)
        self.Conv_04 = self._Make_Layer(block, cfg[4], num_blocks[3], stride=2) 

        self.Linear = nn.Linear(cfg[4],num_classes)

    def _Make_Layer(self,block,planes,num_blocks,stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
            # print(layers)
        return nn.Sequential(*layers)

    def forward(self,x):
        out = F.relu(self.Bn_00(self.Conv_00(x)))

        out = self.Conv_01(out)
        out = self.Conv_02(out)
        out = self.Conv_03(out)
        out = self.Conv_04(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.Linear(out)
        
        return out

def ResNet_half_18():
    return ResNet(Residual_block, [2,2,2,2])

def ResNet_half_34():
    return ResNet(Residual_block, [3,4,6,3])

