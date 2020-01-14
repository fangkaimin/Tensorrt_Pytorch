import torch
import numpy as np
import time
import numpy as np
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch as quantization
from torch2trt import torch2trt
from fp16util import network_to_half
from models import *
from torch2trt import TRTModule
from torch.autograd import Variable
import argparse

parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Test')
parser.add_argument('--batch_size', default=1, type=float, help='batch_size')
parser.add_argument('--network', default = squeezenet(pretrained=False), type=bool,help='load your network')
parser.add_argument('--weight_path', default = './checkpoint/SqueezeNet_Single.pth', type=str,help='load_pth')
parser.add_argument('--losslesseval',default=False, type=bool,help='losslesseval_T?')
parser.add_argument('--trtFP32', default=False, type =bool, help='trt_floating_point')
parser.add_argument('--trtFP16', default=False, type=bool, help='trt_sample_img_batch')
parser.add_argument('--trtchannels', default=3, type=int, help='trt_sample_img_channels')
parser.add_argument('--trtpixels', default=32, type=int, help='trt_sample_img_pixels')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# CIFAR100 data load & transforms
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5070751592371323, 0.48654887331495095, 0.4409178433670343), (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)),
])
testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=4)

if args.losslesseval: 
    if args.trtFP32==True:
        print("losslessFP32")
        net = args.network
        net.load_state_dict(torch.load(args.weight_path))
        net.to(device)
        net.eval()
        cudnn.benuchmark = True
    elif args.trtFP16==True:
        print("losslessFP16")
        net = args.network
        net.load_state_dict(torch.load(args.weight_path))
        net.half().to(device)
        net.eval()
        cudnn.benuchmark = True


def TensorRT32(epoch):
    # make_sample_image
    sample_img = np.random.rand(args.batch_size,args.trtchannels,args.trtpixels,args.trtpixels)
    # dummy_input FP 32
    input = torch.from_numpy(sample_img).view(args.batch_size,args.trtchannels,args.trtpixels,args.trtpixels).float().to(device)
    tensorrt_net = args.network
    tensorrt_net.load_state_dict(torch.load(args.weight_path))
    tensorrt_net.float().to(device)
    # convert trt model FP 32
    net_trt = torch2trt(tensorrt_net,[input],fp16_mode = False,max_batch_size = args.batch_size)
    net_trt.to(device)
    net_trt.eval() 
    print(input.dtype)
    print('=========Start_TensorRT_FP32_Check==========')
    correct = 0
    total = 0
    avg_frame_total = 0.
    TensorRTFP32_time = []
    with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                # convert inputs dtype
                inputs = Variable(inputs).to(device)
                targets = Variable(targets).to(device)
                tic = time.time()
                outputs = net_trt(inputs)
                toc = time.time()

                if batch_idx != 0:
                        TensorRTFP32_time.append(toc-tic)
                        avg_frame_total += (TensorRTFP32_time[-1])
                        avg_frame = avg_frame_total/batch_idx
                        avg_frame = 1/avg_frame
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if ((batch_idx != 0)and((total%2000==0)or (total%10000==1))):
                    print(batch_idx, len(testloader), 'Acc: %.4f%% (%d/%d),TRT Speed is : (%.3f [FPS])'
                    % (100.*correct/total, correct, total, avg_frame))

def TensorRT16(epoch):
    # make_sample_image
    sample_img = np.random.rand(args.batch_size,args.trtchannels,args.trtpixels,args.trtpixels)
    # dummy_input FP 16
    input = torch.from_numpy(sample_img).view(args.batch_size,args.trtchannels,args.trtpixels,args.trtpixels).half().to(device)
    tensorrt_net = args.network
    tensorrt_net.load_state_dict(torch.load(args.weight_path))
    tensorrt_net.half().to(device)
    # convert trt model FP 16
    net_trt = torch2trt(tensorrt_net,[input],fp16_mode = True,max_batch_size = args.batch_size)
    net_trt.to(device)
    net_trt.eval()
    print(input.dtype)
    print('=========Start_TensorRT_FP16_Check==========')
    correct = 0
    total = 0
    avg_frame_total = 0.
    TensorRTFP16_time = []
    with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                # convert inputs dtype

                inputs = Variable(inputs).half().to(device)
                targets = Variable(targets).to(device)
                
                tic = time.time()
                outputs = net_trt(inputs)
                toc = time.time()

                if batch_idx != 0:
                    TensorRTFP16_time.append(toc-tic)
                    avg_frame_total += (TensorRTFP16_time[-1])
                    avg_frame = avg_frame_total/batch_idx
                    avg_frame = 1/avg_frame
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                if ((batch_idx != 0)and((total%2000==0)or (total%10000==1))):
                    print(batch_idx, len(testloader), 'Acc: %.4f%% (%d/%d),TRT Speed is : (%.3f [FPS])'
                    % (100.*correct/total, correct, total, avg_frame))

for epoch in range(1):
    if args.trtFP32 and args.losslesseval:
        TensorRT32(epoch)
    
    if args.trtFP16 and args.losslesseval:    
        TensorRT16(epoch)
    
    if args.trtFP32 and (args.losslesseval == False):
        TensorRT32(epoch)

    if args.trtFP16 and (args.losslesseval == False):
        TensorRT16(epoch)
