
import torch as t
from torch import nn
import os
import torch.nn.functional as F
import numpy as np

#os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
device1 = t.device('cuda:0')
device2 = t.device('cuda:0')

class CONVx2(nn.Module):

    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.conv_x_2 = nn.Sequential(
            nn.Conv3d( in_channels, out_channels, kernel_size = 3, padding=1),
            nn.BatchNorm3d( out_channels, track_running_stats=False),
            nn.LeakyReLU( inplace=True),
            nn.Conv3d( out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d( out_channels, track_running_stats=False),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self,x):
        return self.conv_x_2(x)

class BlurPool(nn.Module): 
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)),int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = t.Tensor(a[:,None]*a[None,:])
        filt = filt/t.sum(filt)
        
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,self.filt_size,1,1)))
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride,::self.stride]
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride,::self.stride]
        else:
            
            #print(self.filt.shape)
            
            return F.conv3d(self.pad(inp), self.filt, stride=self.stride,groups = inp.shape[1])

def get_pad_layer(pad_type):
    if(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad3d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer

class ResnetBlock(nn.Module):

    def __init__(self, in_channels, out_channels,kernelSize=3):
        super().__init__()
        self.resnetblock1 = nn.Sequential(
            nn.Conv3d(in_channels,out_channels,kernel_size=kernelSize,padding=1),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.LeakyReLU(inplace=True),
            )
        self.resnetblock2 = nn.Sequential(
            nn.Conv3d(out_channels,out_channels,kernel_size=kernelSize,padding=1),
            nn.BatchNorm3d(out_channels, track_running_stats=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels,out_channels,kernel_size=kernelSize,padding=1),
            nn.BatchNorm3d(out_channels, track_running_stats=False)
            )

        self.reluUnit = nn.Sequential(
            nn.LeakyReLU(inplace=True)
            )

    def forward(self,x):
        x=self.resnetblock1(x)
        x1=x
        x=self.resnetblock2(x)
        #x = self.se(x)
        x=x+x1
        x=self.reluUnit(x)
        return x

class Downsampling(nn.Module):

    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.downstep = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            BlurPool(in_channels,'repl',filt_size=4,stride=2),
            #CONVx2(in_channels,out_channels)
            ResnetBlock(in_channels,out_channels),
            #ResnetBlock(out_channels,out_channels)
        )
    
    def forward(self,x):
        return self.downstep(x)



class GridAttentionGateLocal3D(nn.Module):

    def __init__(self, Fg, Fl, Fint, learn_upsampling=False, batchnorm=True):
        super(GridAttentionGateLocal3D, self).__init__()

        if batchnorm:
            self.Wg = nn.Sequential(
                nn.Conv3d(Fg, Fint, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm3d(Fint, track_running_stats=False)
            )
            self.Wx = nn.Sequential(
                nn.Conv3d(Fl, Fint, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(Fint, track_running_stats=False),
                nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                BlurPool(Fint,'repl',filt_size=4,stride=2)
            )

            self.y = nn.Sequential(
                nn.Conv3d(in_channels=Fint, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm3d(1, track_running_stats=False)
            )

        else:
            self.Wg = nn.Conv3d(Fg, Fint, kernel_size=1, stride=1, padding=0, bias=True)
            self.Wx = nn.Sequential(
                nn.Conv3d(Fl, Fint, kernel_size=1, stride=1, padding=0, bias=False),
                nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
                BlurPool(Fint,'repl',filt_size=4,stride=2)
            )

            self.y = nn.Conv3d(in_channels=Fint, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)

        self.out = nn.Sequential(
            nn.Conv3d(in_channels=Fl, out_channels=Fl, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(Fl, track_running_stats=False),
        )

    def forward(self, xl, g):

        xl_size_orig = xl.size()
        xl_ = self.Wx(xl)

        g = self.Wg(g)

        relu = F.relu(xl_ + g, inplace=True)
        y = self.y(relu)
        sigmoid = t.sigmoid(y)

        upsampled_sigmoid = F.interpolate(sigmoid, size=xl_size_orig[2:], mode='trilinear', align_corners=False)

        # scale features with attention
        attention = upsampled_sigmoid.expand_as(xl)

        return self.out(attention * xl)


class Upsampling(nn.Module):

    def __init__(self,in_channels,out_channels): #can try out bilinear upsampling too instead
        super().__init__()
        self.att = GridAttentionGateLocal3D(Fg=in_channels, Fl=out_channels, Fint=out_channels)
        self.upsampled = nn.ConvTranspose3d(in_channels,out_channels,kernel_size=2,stride=2)
        #self.conv3d = CONVx2(out_channels*2,out_channels)
        self.conv3d = nn.Sequential(
            ResnetBlock(out_channels*2,out_channels),
            #ResnetBlock(out_channels,out_channels)
        )
    def forward(self,x1,x2):
        g3 = self.att(x2, x1)
        x1 = self.upsampled(x1)
        # input is CHW
        
        diffZ = g3.size()[2] - x1.size()[2]
        diffY = g3.size()[3] - x1.size()[3]
        diffX = g3.size()[4] - x1.size()[4]
        x1 = F.pad(x1, [diffX//2, diffX -( diffX//2), diffY//2, diffY-(diffY//2) , diffZ//2, diffZ-(diffZ//2)])
        
        x = t.cat([g3, x1], dim=1)

        return self.conv3d(x)

class OutputConv(nn.Module):
    def __init__(self,in_channels,out_classes):
        super(OutputConv,self).__init__()
        self.conv = nn.Conv3d(in_channels, out_classes, kernel_size=1)
    
    def forward(self,x):
        return self.conv(x)

class UnetModel(nn.Module):
    def __init__(self,in_channels,out_classes):
        super(UnetModel,self).__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes

        self.cv1 = ResnetBlock(in_channels,32).to(device1)
        self.down1 = Downsampling(32,64).to(device1)
        self.down2 = Downsampling(64,128).to(device1)
        self.down3 = Downsampling(128,256).to(device1)
        
        self.up1 = Upsampling(256,128).to(device1)
        self.up2 = Upsampling(128,64).to(device1)
        self.up3 = Upsampling(64,32).to(device2)
        self.cv2 = OutputConv(32,out_classes).to(device1)
        self.initialize()
    
    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self,x):
        x1 = self.cv1(x.to(device1))
        x2 = self.down1(x1.to(device1))
        x3 = self.down2(x2.to(device1))
        x4 = self.down3(x3.to(device1))
        
        x = self.up1(x4.to(device1), x3.to(device1))
        x = self.up2(x.to(device1), x2.to(device1))
        x = self.up3(x.to(device2), x1.to(device2))
        logits = self.cv2(x.to(device1))
        return logits
