import torch
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms as standard_transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from PIL import Image 
import matplotlib.pyplot as plt
import numpy as np
from torch.autograd import Variable
import sys
from models import segnet_modified_with_skip as segnet_s
from models import segnet_modified as segnet_mo
from models import segnet as segnet_b
DATA_PATH = '/home/bala/Machine Learning/Test_project/'
train = datasets.Cityscapes(DATA_PATH, split = 'train', mode = 'fine', target_type = 'semantic',transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]),target_transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]))
test = datasets.Cityscapes(DATA_PATH, split = 'test', mode = 'fine', target_type = 'semantic' ,transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]),target_transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]))
val = datasets.Cityscapes(DATA_PATH, split = 'val', mode = 'fine', target_type = 'semantic' ,transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]),target_transform=transforms.Compose([transforms.Resize((256,512)),transforms.ToTensor()]))

trainset = torch.utils.data.DataLoader(train, batch_size=2, shuffle=True)
testset = torch.utils.data.DataLoader(test, batch_size=2, shuffle=False)
valset = torch.utils.data.DataLoader(val, batch_size=2, shuffle=True)

def main():
  if torch.cuda.is_available():
    device = torch.device("cuda:0")  # you can continue going on here, like cuda:1 cuda:2....etc. 
    print("Running on the GPU")
  else:
    device = torch.device("cpu")
    print("Running on the CPU")
  net = segnet_b.SegNet_sequential().to(device)               # segnet_b.SegNet_sequential().to(device) for loading SegNet base architecture
  print(net)                                                  # segnet_mo.SegNet_sequential().to(device) for loading modified SegNet architecture
  return net,device                                           # segnet_s.SegNet_sequential().to(device) for loading modified SegNet with Skip connections
       
def train(net, device):
  weight = torch.ones(34)
  loss_function = nn.CrossEntropyLoss(weight).cuda()
  optimizer = optim.Adam(net.parameters(), lr=0.01)
  for epoch in range(100): 
    for data in trainset:  
        X, y = data
        X, y = X.to(device), y.to(device)
        net.zero_grad()  
        output = net(X)
        output = output.view(output.size(0),output.size(1), -1)     # For converting the result into a single column vector
        output = torch.transpose(output,1,2).contiguous()
        output = output.view(-1,output.size(2))                 
        label = y*255
        label = label.long()
        label = label.view(-1)
        loss = loss_function(output, label)
        loss.backward() 
        optimizer.step()
    print("Epoch No:",epoch)
    print(loss) 
    torch.save(net.state_dict(),'/home/bala/Machine Learning/Test_project/wts_segnet.pth')  #For saving weights after every Epoch 

def decode_segmap(image, nc=31):
   
  label_colors = np.array([(0, 0, 0),  
    
               (128, 0, 0), (128,64,128), (128, 128, 0), (0, 0, 50), (128, 0, 128),

               (0, 128, 64), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),

               (198, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),

               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
               
               (0,135,0),(128,192,128),(64,128,192),(220,20,60),(64,192,128),
               
               (0, 0,190),(128,128,192),(128,192,64),(128,64,192),(192,64,128)])
 
  r = np.zeros_like(image).astype(np.uint8)
  g = np.zeros_like(image).astype(np.uint8)
  b = np.zeros_like(image).astype(np.uint8)
   
  for l in range(0, nc):
    idx = image == l
    r[idx] = label_colors[l, 0]
    g[idx] = label_colors[l, 1]
    b[idx] = label_colors[l, 2]
     
  rgb = np.stack([r, g, b], axis=2)
  return rgb


def test(net, device):
  correct = 0
  valset_size = 500
  print("Testing")
  with torch.no_grad():
      nonzerocount = 0
      for data in valset:
          X2, y = data
          X, y = X2.to(device), y.to(device)
          output = net(X)
          for idx in range(len(output)):
            out = output[idx].cpu().max(0)[1].data.squeeze(0).byte().numpy()
            predicted_mx = output[idx]
            predicted_mx_idx = torch.argmax(predicted_mx,0)
            predicted_mx_idx = predicted_mx_idx.detach().cpu().numpy()            # finding class index with maximum softmax probability
            rgb = decode_segmap(predicted_mx_idx)
            fig = plt.figure(1)
            plt.imshow(rgb)
            plt.figure(2)
            plt.imshow(transforms.ToPILImage()(data[0][idx]))#.detach().cpu().numpy())
            plt.show()
            label = y[idx][0].detach().cpu().numpy()
            final_diff = predicted_mx_idx - label*255
            nonzerocount = nonzerocount + np.count_nonzero(final_diff)
      accu = 1 - nonzerocount/(valset_size*256*512)
      print("Accuracy",accu)

def load_pretrained_weights(net):
  vgg = torchvision.models.vgg16(pretrained=True,progress=True)
  pretrained_dict = vgg.state_dict()
  model_dict = net.state_dict()
  list1 = ['layer10_conv.weight',
    'layer10_conv.bias',
    'layer11_conv.weight',
    'layer11_conv.bias',
    'layer20_conv.weight',
    'layer20_conv.bias',
    'layer21_conv.weight',
    'layer21_conv.bias',
    'layer30_conv.weight',
    'layer30_conv.bias',
    'layer31_conv.weight',
    'layer31_conv.bias',
    'layer32_conv.weight',
    'layer32_conv.bias',
    'layer40_conv.weight',
    'layer40_conv.bias',
    'layer41_conv.weight',
    'layer41_conv.bias',
    'layer42_conv.weight',
    'layer42_conv.bias',
    'layer50_conv.weight',
    'layer50_conv.bias',
    'layer51_conv.weight',
    'layer51_conv.bias',
    'layer52_conv.weight',
    'layer52_conv.bias'
    ]
  list2 = ['features.0.weight',
    'features.0.bias',
    'features.2.weight',
    'features.2.bias',
    'features.5.weight',
    'features.5.bias',
    'features.7.weight',
    'features.7.bias',
    'features.10.weight',
    'features.10.bias',
    'features.12.weight',
    'features.12.bias',
    'features.14.weight',
    'features.14.bias',
    'features.17.weight',
    'features.17.bias',
    'features.19.weight',
    'features.19.bias',
    'features.21.weight',
    'features.21.bias',
    'features.24.weight',
    'features.24.bias',
    'features.26.weight',
    'features.26.bias',
    'features.28.weight',
    'features.28.bias'
    ]
  for l in range(len(list1)):
    pretrained_dict[list1[l]] = pretrained_dict.pop(list2[l])

  pretrained_dict = {k: v for k,v in pretrained_dict.items() if k in model_dict}
  model_dict.update(pretrained_dict)
  net.load_state_dict(model_dict)
  return net

if __name__ == '__main__':
  sec = time.time()
  net, device = main()
  net = load_pretrained_weights(net)                   # loading pretrained weights of vgg16 network for the encoders of SegNet
  # net.load_state_dict(torch.load('/home/bala/Machine Learning/Test_project/wts_segnet.pth'))          # For loading the learned weights for validation and testing
  train(net,device)                                  # For training the network
  test(net, device)                                    # For testing the network
  sec_last = time.time()
  print("time",sec_last-sec)
