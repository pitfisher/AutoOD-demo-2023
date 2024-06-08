import torch.nn as nn
import torch
from torchvision import transforms
from torchvision.ops import nms
from retinanet.utils import BBoxTransform, ClipBoxes, AnchorsP2





class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


    
class CustomPyramidFeatures_inception_concat_p2(nn.Module):
    def __init__(self, C2_size, C3_size, C4_size, C5_size, layers = [3,4,5,6,7], feature_size=256):
        super(CustomPyramidFeatures_inception_concat_p2, self).__init__()
        
        self.layers = layers

        # upsample C5 to get P5 from the FPN paper
        self.P5_1_1 = nn.Conv2d(C5_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        # self.P5_1_3 = nn.Conv2d(C5_size, feature_size//2, kernel_size=3, stride=1, padding=1)
        
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size//2, feature_size, kernel_size=3, stride=1, padding=1)
        

        
        self.P4_1_1 = nn.Conv2d(C4_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        # self.P4_1_3 = nn.Conv2d(C4_size, feature_size//2, kernel_size=3, stride=1, padding=1)
        
        self.P4_12_1 = nn.Conv2d(feature_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        

        
        self.P3_1_1 = nn.Conv2d(C3_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        self.P3_1_3 = nn.Conv2d(C3_size, feature_size//2, kernel_size=3, stride=1, padding=1)
        
        self.P3_12_1 = nn.Conv2d(feature_size, feature_size//2, kernel_size=1, stride=1, padding=0)

        self.P3_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        

        self.P2_1_1 = nn.Conv2d(C2_size, feature_size//2, kernel_size=1, stride=1, padding=0)
        self.P2_1_3 = nn.Conv2d(C2_size, feature_size//2, kernel_size=3, stride=1, padding=1)

        self.P2_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)
        


    def forward(self, C2, C3, C4, C5):

        P5_x = self.P5_1_1(C5) #

        P5_upsampled_x = self.P5_upsampled(P5_x)
        # P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1_1(C4) #

        P4_x = torch.cat((P5_upsampled_x,P4_x),dim=1) #P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_12_1(self.P4_upsampled(P4_x))
        # P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1_1(C3) #
        P3_x = P3_x + self.P3_1_3(C3)
        P3_x = torch.cat((P4_upsampled_x,P3_x),dim=1) #P3_x + P4_upsampled_x
        P3_upsampled_x = self.P3_12_1(self.P3_upsampled(P3_x))
        P3_x = self.P3_2(P3_x)

        P2_x = self.P2_1_1(C2) #
        P2_x = P2_x + self.P2_1_3(C2)
        P2_x = torch.cat((P3_upsampled_x,P2_x),dim=1) #P3_x + P4_upsampled_x
        P2_x = self.P2_2(P2_x)
        
             
        return P2_x, P3_x
    



class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, inputs = 3, fpn = CustomPyramidFeatures_inception_concat_p2):
        self.inplanes = 64
        self.num_classes = num_classes
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(inputs, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        fpn_sizes = [self.layer2[layers[1] - 1].conv3.out_channels, self.layer3[layers[2] - 1].conv3.out_channels,
                         self.layer4[layers[3] - 1].conv3.out_channels]

        self.fpn = fpn(fpn_sizes[0]//2, fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.regressionModel = RegressionModel(256)
        self.classificationModel = ClassificationModel(256, num_classes=self.num_classes)

        




    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

  

    def forward(self, inputs):
        
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        

        p2, p3 = self.fpn(x1, x2, x3, x4)

        regression = torch.cat([self.regressionModel(p2), self.regressionModel(p3)], dim=1)
        classification = torch.cat([self.classificationModel(p2), self.classificationModel(p3)], dim=1)
        
        
        return classification, regression
        
        


def RetinaNet(num_classes, inputs = 3, **kwargs):

    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], inputs = inputs, **kwargs)

    return model



class ModelWrapper(nn.Module):
    def __init__(self, model, img_shape = [1024,1024]):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.img_shape = img_shape

        self.preprocess = transforms.Compose([
            transforms.Resize((self.img_shape[0], self.img_shape[1])),  
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225]),
        ])
        
        self.regressBoxes = BBoxTransform()
        self.clipBoxes = ClipBoxes()
        self.anchors = AnchorsP2([2,3]).forward(torch.zeros(1,3,self.img_shape[0],self.img_shape[1]))
        

    def forward(self, x):
        _,_, H, W = x.shape
        classification, regression = self.model(self.preprocess(x))

        anchorBoxes = self.regressBoxes(self.anchors, regression[:,:self.anchors.shape[1],:])
        anchorBoxes = self.clipBoxes(anchorBoxes, self.img_shape)

        scores = torch.squeeze(classification[:, :self.anchors.shape[1], 1])
        scores_over_thresh = (scores > 0.5)
        if scores_over_thresh.sum() == 0:
            scores = torch.tensor([])
            boxes = torch.tensor([])
        else:
            scores = scores[scores_over_thresh]
            anchorBoxes = torch.squeeze(anchorBoxes)
            anchorBoxes = anchorBoxes[scores_over_thresh]
            anchors_nms_idx = nms(anchorBoxes, scores, 0.1)
            scores = scores[anchors_nms_idx]
            boxes = anchorBoxes[anchors_nms_idx]
            boxes[:,0] = boxes[:,0] * W / self.img_shape[1]
            boxes[:,1] = boxes[:,1] * H / self.img_shape[0]
            boxes[:,2] = boxes[:,2] * W / self.img_shape[1]
            boxes[:,3] = boxes[:,3] * H / self.img_shape[0]
            
                
        return [{'boxes': boxes,'scores': scores}]
    





