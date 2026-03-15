import torch
import torch.nn as nn
import torch.nn.functional as F

BLOCKS = {


    "conv_3x3": lambda C_in, C_out, stride:
        Conv3x3(C_in, C_out, stride),

    "dil_conv_3x3": lambda C_in, C_out, stride:
        DilConv(C_in, C_out, stride),

    "depthwise_sep_conv": lambda C_in, C_out, stride:
        DepthwiseSeparableConv(C_in, C_out, stride),

    "sep_conv": lambda C_in, C_out, stride:
        SepConv(C_in, C_out, stride),

    "inverted_residual": lambda C_in, C_out, stride:
        InvertedResidual(C_in, C_out, stride),

    "basic_block": lambda C_in, C_out, stride:
        BasicBlock(C_in, C_out, stride),

    "bottleneck": lambda C_in, C_out, stride:
        Bottleneck(C_in, C_out, stride),

    "factorized_reduce": lambda C_in, C_out, stride:
        FactorizedReduce(C_in, C_out)
}

class Conv3x3(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.out_ch = out_ch
        self.stride = stride
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)

class DilConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, dilation=2):
        super().__init__()

        self.dw = nn.Conv2d(
            in_ch, in_ch,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            groups=in_ch,
            bias=False
        )
        self.out_ch = out_ch
        self.stride = stride
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.dw(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pw(x)
        x = self.bn2(x)

        return self.relu2(x)
    
class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.out_ch = out_ch
        self.stride = stride

        self.dw = nn.Conv2d(
            in_ch, in_ch,
            3, stride, 1,
            groups=in_ch,
            bias=False
        )

        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.dw(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pw(x)
        x = self.bn2(x)

        return self.relu2(x)
    
class SepConv(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        self.out_ch = out_ch
        self.stride = stride
        self.dw1 = nn.Conv2d(in_ch, in_ch, 3, stride, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.pw1 = nn.Conv2d(in_ch, in_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_ch)
        self.relu2 = nn.ReLU(inplace=True)

        self.dw2 = nn.Conv2d(in_ch, in_ch, 3, 1, 1, groups=in_ch, bias=False)
        self.bn3 = nn.BatchNorm2d(in_ch)
        self.relu3 = nn.ReLU(inplace=True)

        self.pw2 = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(out_ch)
        self.relu4 = nn.ReLU(inplace=True)

    def forward(self, x):

        x = self.dw1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pw1(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.dw2(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.pw2(x)
        x = self.bn4(x)

        return self.relu4(x)

class InvertedResidual(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1, use_residual=True):
        super().__init__()
        self.out_ch = out_ch
        self.stride = stride

        self.hidden = int(in_ch * 6)
        self.use_residual = use_residual

        self.expand = nn.Conv2d(in_ch, self.hidden, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.hidden)
        self.relu1 = nn.ReLU6(inplace=True)

        self.dw = nn.Conv2d(self.hidden, self.hidden, 3, stride, 1, groups=self.hidden, bias=False)
        self.bn2 = nn.BatchNorm2d(self.hidden)
        self.relu2 = nn.ReLU6(inplace=True)

        self.project = nn.Conv2d(self.hidden, out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)

    def forward(self, x):

        identity = x

        x = self.expand(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.dw(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.project(x)
        x = self.bn3(x)

        if self.use_residual:

        ################# This part is adjusted for channel pruning #############
            if x.shape!=identity.shape:
                if x.size(1)<identity.size(1):
                    identity=identity[:,:x.size(1),:,:]
                else:
                    padded = torch.zeros_like(x)
                    padded[:,:identity.size(1),:,:]=identity
                    identity=padded
        #########################################################################
            x = x + identity

        return x
    
class BasicBlock(nn.Module):

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.out_ch = out_ch
        self.stride = stride

        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )

    def forward(self, x):

        identity = x
        print("Input shape:", x.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        print("After conv1:", out.shape)
        out = self.conv2(out)
        out = self.bn2(out)
        print("After conv2:", out.shape)
        if self.downsample is not None:
            identity = self.downsample(x)
            print("After downsample:", identity.shape)
        ################# This part is adjusted for channel pruning #############
            # if x.shape!=identity.shape:
            #     if x.size(1)<identity.size(1):
            #         identity=identity[:,:x.size(1),:,:]
            #     else:
            #         padded = torch.zeros_like(x)
            #         padded[:,:identity.size(1),:,:]=identity
            #         identity=padded
        #########################################################################
    
        out += identity

        return self.relu(out)


class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()

        hidden_channels = out_ch
        self.out_ch = out_ch*4
        self.stride = stride

        self.conv1 = nn.Conv2d(in_ch, hidden_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, 3, stride, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels)

        self.conv3 = nn.Conv2d(hidden_channels, self.out_ch, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.out_ch)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(self.out_ch)
            )

    def forward(self, x):

        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)

        if self.downsample is not None:
            identity = self.downsample(identity)
        ################# This part is adjusted for channel pruning #############
            if x.shape!=identity.shape:
                if x.size(1)<identity.size(1):
                    identity=identity[:,:x.size(1),:,:]
                else:
                    padded = torch.zeros_like(x)
                    padded[:,:identity.size(1),:,:]=identity
                    identity=padded
        #########################################################################

        x += identity

        return self.relu(x)
    
class FactorizedReduce(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()

        assert out_ch % 2 == 0
        self.out_ch = out_ch

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=2, bias=False)
        self.conv2 = nn.Conv2d(in_ch, out_ch // 2, 1, stride=2, bias=False)

        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):

        x = self.relu(x)

        out = torch.cat([
            self.conv1(x),
            self.conv2(x[:, :, 1:, 1:])
        ], dim=1)

        return self.bn(out)