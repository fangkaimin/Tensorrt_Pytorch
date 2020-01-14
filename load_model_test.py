import torch
import numpy as np
import time
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from torch2trt import torch2trt
from fp16util import network_to_half
from models import *
from torch2trt import TRTModule
from torch.autograd import Variable
import copy
import argparse



device = 'cuda' if torch.cuda.is_available() else 'cpu'

#load model
print('==> Building Network..')
net = ResNet_half_18()
#net = net.to(device)
cudnn.benuchmark = True
net.load_state_dict(torch.load('./checkpoint/ResNet18_Single.pth'))
net.to(device)

fff