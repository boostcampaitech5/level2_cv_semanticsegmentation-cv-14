import torch.nn as nn
import torch
import torch.nn.functional as F

class FCN32s(nn.Module):
    def __init__(self, num_classes=29): # 29
        super(FCN32s, self).__init__()
        
        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride, 
                                            padding=padding),
                                  nn.ReLU(inplace=True)
                                 )
    
        # conv1
        self.conv1_1 = CBR(3, 64, 3, 1, 1)
        self.conv1_2 = CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 
        
        # conv2
        self.conv2_1 = CBR(64, 128, 3, 1, 1)
        self.conv2_2 = CBR(128, 128, 3, 1, 1)  
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 

        # conv3
        self.conv3_1 = CBR(128, 256, 3, 1, 1)
        self.conv3_2 = CBR(256, 256, 3, 1, 1)
        self.conv3_3 = CBR(256, 256, 3, 1, 1)          
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 

        # conv4
        self.conv4_1 = CBR(256, 512, 3, 1, 1)
        self.conv4_2 = CBR(512, 512, 3, 1, 1)
        self.conv4_3 = CBR(512, 512, 3, 1, 1)          
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        # conv5
        self.conv5_1 = CBR(512, 512, 3, 1, 1)
        self.conv5_2 = CBR(512, 512, 3, 1, 1)
        self.conv5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        
        # fc6
        self.fc6 = CBR(512, 4096, 1, 1, 0)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = CBR(4096, 4096, 1, 1, 0)
        self.drop7 = nn.Dropout2d()

        # Score 
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size=1, stride=1, padding=0)
        
        # UPScore using deconv
        self.upscore32 = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16)

                    
    def forward(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h = self.pool1(h)

        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = self.pool2(h)

        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)        
        h = self.pool3(h)

        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)        
        h = self.pool4(h)

        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)        
        h = self.pool5(h)
        
        h = self.fc6(h)
        h = self.drop6(h)

        h = self.fc7(h)
        h = self.drop7(h)
        
        h = self.score_fr(h)
        upscore32 = self.upscore32(h)  
        
        return upscore32
    

# 모델 참고 코드 
# https://github.com/wkentaro/pytorch-fcn/

class FCN16s(nn.Module):
    def __init__(self, num_classes=29):
        super(FCN16s, self).__init__()
        self.relu    = nn.ReLU(inplace=True)
        
        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride, 
                                            padding=padding),
                                  nn.ReLU(inplace=True)
                                 )
        
        # conv1
        self.conv1_1 = CBR(3, 64, 3, 1, 1)
        self.conv1_2 = CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 
        
        # conv2
        self.conv2_1 = CBR(64, 128, 3, 1, 1)
        self.conv2_2 = CBR(128, 128, 3, 1, 1)  
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 

        # conv3
        self.conv3_1 = CBR(128, 256, 3, 1, 1)
        self.conv3_2 = CBR(256, 256, 3, 1, 1)
        self.conv3_3 = CBR(256, 256, 3, 1, 1)          
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 

        # conv4
        self.conv4_1 = CBR(256, 512, 3, 1, 1)
        self.conv4_2 = CBR(512, 512, 3, 1, 1)
        self.conv4_3 = CBR(512, 512, 3, 1, 1)          
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Score pool4
        self.score_pool4_fr = nn.Conv2d(512,
                                        num_classes, 
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)        
        
        # conv5
        self.conv5_1 = CBR(512, 512, 3, 1, 1)
        self.conv5_2 = CBR(512, 512, 3, 1, 1)
        self.conv5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
    
        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        # Score
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size = 1)
        
        # UpScore2 using deconv
        self.upscore2 = nn.ConvTranspose2d(num_classes,
                                           num_classes,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)
        
        # UpScore16 using deconv
        self.upscore16 = nn.ConvTranspose2d(num_classes, 
                                            num_classes, 
                                            kernel_size=32,
                                            stride=16,
                                            padding=8)
        

    def forward(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h = self.pool1(h)

        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = self.pool2(h)

        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)        
        h = self.pool3(h)

        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)        
        pool4 = h = self.pool4(h)

        # Score
        score_pool4c = self.score_pool4_fr(pool4)
        
        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)        
        h = self.pool5(h)
        
        h = self.fc6(h)
        h = self.relu6(h) # update
        h = self.drop6(h)

        h = self.fc7(h)
        h = self.relu7(h) # update
        h = self.drop7(h)
        
        h = self.score_fr(h)
        
        # Up Score I       
        upscore2 = self.upscore2(h)
        
        # Sum I
        h = upscore2 + score_pool4c
        
        # Up Score II
        upscore16 = self.upscore16(h)

        return upscore16

# 모델 참고 코드 
# https://github.com/wkentaro/pytorch-fcn/

class FCN8s(nn.Module):
    def __init__(self, num_classes=29):
        super(FCN8s, self).__init__()
        
        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            return nn.Sequential(nn.Conv2d(in_channels=in_channels, 
                                            out_channels=out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride, 
                                            padding=padding),
                                  nn.ReLU(inplace=True)
                                 )        
        
        # conv1
        self.conv1_1 = CBR(3, 64, 3, 1, 1)
        self.conv1_2 = CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 
        
        # conv2
        self.conv2_1 = CBR(64, 128, 3, 1, 1)
        self.conv2_2 = CBR(128, 128, 3, 1, 1)  
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 

        # conv3
        self.conv3_1 = CBR(128, 256, 3, 1, 1)
        self.conv3_2 = CBR(256, 256, 3, 1, 1)
        self.conv3_3 = CBR(256, 256, 3, 1, 1)          
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True) 

        # Score pool3
        self.score_pool3_fr = nn.Conv2d(256,
                                        num_classes, 
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)             
        
        # conv4
        self.conv4_1 = CBR(256, 512, 3, 1, 1)
        self.conv4_2 = CBR(512, 512, 3, 1, 1)
        self.conv4_3 = CBR(512, 512, 3, 1, 1)          
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Score pool4
        self.score_pool4_fr = nn.Conv2d(512,
                                        num_classes, 
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)        
        
        # conv5
        self.conv5_1 = CBR(512, 512, 3, 1, 1)
        self.conv5_2 = CBR(512, 512, 3, 1, 1)
        self.conv5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
    
        # fc6
        self.fc6 = nn.Conv2d(512, 4096, 1)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        # fc7
        self.fc7 = nn.Conv2d(4096, 4096, 1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()


        # Score
        self.score_fr = nn.Conv2d(4096, num_classes, kernel_size = 1)
        
        
        # UpScore2 using deconv
        self.upscore2 = nn.ConvTranspose2d(num_classes,
                                           num_classes,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1)
        
        # UpScore2_pool4 using deconv
        self.upscore2_pool4 = nn.ConvTranspose2d(num_classes, 
                                                 num_classes, 
                                                 kernel_size=4,
                                                 stride=2,
                                                 padding=1)
        
        # UpScore8 using deconv
        self.upscore8 = nn.ConvTranspose2d(num_classes, 
                                           num_classes,
                                           kernel_size=16,
                                           stride=8,
                                           padding=4)


    def forward(self, x):
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h = self.pool1(h)

        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h = self.pool2(h)

        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)        
        pool3 = h = self.pool3(h)

        # Score
        score_pool3c = self.score_pool3_fr(pool3)

        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)        
        pool4 = h = self.pool4(h)

        # Score
        score_pool4c = self.score_pool4_fr(pool4)       
        
        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)        
        h = self.pool5(h)
        
        h = self.fc6(h)
        h = self.relu6(h) # update
        h = self.drop6(h)

        h = self.fc7(h)
        h = self.relu7(h) # update
        h = self.drop7(h)
        
        h = self.score_fr(h)
        
        # Up Score I
        upscore2 = self.upscore2(h)
        
        # Sum I
        h = upscore2 + score_pool4c
        
        # Up Score II
        upscore2_pool4c = self.upscore2_pool4(h)
        
        # Sum II
        h = upscore2_pool4c + score_pool3c
        
        # Up Score III
        upscore8 = self.upscore8(h)
        
        return upscore8




def conv_block(in_ch, out_ch, k_size, stride, padding, dilation=1, relu=True):
    block = []
    block.append(nn.Conv2d(in_ch, out_ch, k_size, stride, padding, dilation, bias=False))
    block.append(nn.BatchNorm2d(out_ch))
    if relu:
        block.append(nn.ReLU())
    return nn.Sequential(*block)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation=1):
        super().__init__()
        if dilation > kernel_size//2:
            padding = dilation
        else:
            padding = kernel_size//2
            
        self.depthwise_conv = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride, padding,
            dilation=dilation, groups=in_ch, bias=False
        )
        self.pointwise_conv = nn.Conv2d(
            in_ch, out_ch, 1, 1, bias=False
        )
        self.bn = nn.BatchNorm2d(in_ch)
        
    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.bn(out)
        out = self.pointwise_conv(out)
        return out


class XceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, exit_flow=False, use_1st_relu=True):
        super().__init__()
        if in_ch != out_ch or stride !=1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else: 
            self.skip = None
        
        if exit_flow:
            block = [
                nn.ReLU(),
                DepthwiseSeparableConv2d(in_ch, in_ch, 3, 1, dilation),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                DepthwiseSeparableConv2d(in_ch, out_ch, 3, 1, dilation),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                DepthwiseSeparableConv2d(out_ch, out_ch, 3, stride, dilation),
                nn.BatchNorm2d(out_ch) 
            ]
        else:
            block = [
                nn.ReLU(),
                DepthwiseSeparableConv2d(in_ch, out_ch, 3, 1, dilation),
                nn.BatchNorm2d(out_ch),            
                nn.ReLU(),
                DepthwiseSeparableConv2d(out_ch, out_ch, 3, 1, dilation),
                nn.BatchNorm2d(out_ch),            
                nn.ReLU(),
                DepthwiseSeparableConv2d(out_ch, out_ch, 3, stride, dilation),
                nn.BatchNorm2d(out_ch)                
            ]
   
        if not use_1st_relu: 
            block = block[1:]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        output = self.block(x)
        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x

        x = output + skip
        return x
    
    
class Xception(nn.Module):
    def __init__(self, in_channels):
        super(Xception, self).__init__()        
        self.entry_block_1 = nn.Sequential(
            conv_block(in_channels, 32, 3, 2, 1),
            conv_block(32, 64, 3, 1, 1, relu=False),
            XceptionBlock(64, 128, 2, 1, use_1st_relu=False)
        )
        self.relu = nn.ReLU()
        self.entry_block_2 = nn.Sequential(
            XceptionBlock(128, 256, 2, 1),
            XceptionBlock(256, 728, 2, 1)
        )
        
        middle_block = [XceptionBlock(728, 728, 1, 1) for _ in range(16)]
        self.middle_block = nn.Sequential(*middle_block)
        
        self.exit_block = nn.Sequential(
            XceptionBlock(728, 1024, 1, 1, exit_flow=True),
            nn.ReLU(),
            DepthwiseSeparableConv2d(1024, 1536, 3, 1, 2),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            DepthwiseSeparableConv2d(1536, 1536, 3, 1, 2),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            DepthwiseSeparableConv2d(1536, 2048, 3, 1, 2),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
        )
            
    def forward(self, x):
        out = self.entry_block_1(x)
        features = out
        out = self.entry_block_2(out)
        out = self.middle_block(out)
        out = self.exit_block(out)
        return out, features
    
    
class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.block1 = conv_block(in_ch, 256, 1, 1, 0, 1)
        self.block2 = conv_block(in_ch, 256, 3, 1, 6, 6)
        self.block3 = conv_block(in_ch, 256, 3, 1, 12, 12)
        self.block4 = conv_block(in_ch, 256, 3, 1, 18, 18)
        self.block5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv = conv_block(256*5, 256, 1, 1, 0)
         
    def forward(self, x):
        upsample_size = (x.shape[-1], x.shape[-2])
        
        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        out5 = self.block5(x)
        out5 = F.interpolate(
            out5, size=upsample_size, mode="bilinear", align_corners=True
        )
        
        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        out = self.conv(out)
        return out


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = conv_block(128, 48, 1, 1, 0)
        self.block2 = nn.Sequential(
            conv_block(48+256, 256, 3, 1, 1),
            conv_block(256, 256, 3, 1, 1),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x, features):
        features = self.block1(features)
        feature_size = (features.shape[-1], features.shape[-2])
        
        out = F.interpolate(x, size=feature_size, mode="bilinear", align_corners=True)
        out = torch.cat((features, out), dim=1)
        out = self.block2(out)
        return out

    
class DeepLabV3p(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.backbone = Xception(in_channels)
        self.aspp = AtrousSpatialPyramidPooling(2048)
        self.decoder = Decoder(num_classes)
        
    def forward(self, x):
        upsample_size = (x.shape[-1], x.shape[-2])

        backbone_out, features = self.backbone(x)
        aspp_out = self.aspp(backbone_out)
        
        out = self.decoder(aspp_out, features)
        out = F.interpolate(
            out, size=upsample_size, mode="bilinear", align_corners=True
        )
        return out
    
# LR = 1e-4
# model = DeepLabV3p(in_channels=3, num_classes=len(CLASSES))

# # Optimizer 정의
# optimizer = optim.SGD(params=model.parameters(), lr=LR, weight_decay=1e-6)
if __name__ == "__main__":
    
    x = torch.randn(8,3,512,512)
    model1 = FCN8s()
    y1 = model1(x)
    model2 = FCN16s()
    y2 = model2(x)
    model3 = FCN32s()
    y3 = model3(x)
    model4 = DeepLabV3p(3,19)
    y4 = model4(x)

    print(x.shape)
    print(y1.shape)
    print(y2.shape)
    print(y3.shape)
    print(y4.shape)