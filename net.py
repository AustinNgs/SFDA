from easydl import *
from torchvision import models
import torch.nn.functional as F
import sys
import pnasnet_model
import nasnet_model

sys.path[0] = '/home/lab-wu.shibin/dann'

class BaseFeatureExtractor(nn.Module):
    def forward(self, *input):
        pass

    def __init__(self):
        super(BaseFeatureExtractor, self).__init__()

    def output_num(self):
        pass

    def train(self, mode=True):
        # freeze BN mean and std
        for module in self.children():
            if isinstance(module, nn.BatchNorm2d):
                module.train(False)
            else:
                module.train(mode)


class ResNet50Fc(BaseFeatureExtractor):
    """
    ** input image should be in range of [0, 1]**
    """
    def __init__(self,model_path=None, normalize=True):
        super(ResNet50Fc, self).__init__()
        #如果本地有自己的resnet，则调用本地的resnet.pkl
        if model_path:
            if os.path.exists(model_path):
                self.model_resnet = models.resnet50(pretrained=False)
                self.model_resnet.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        #否则调用网上pretrain好的resnet，联网下载
        else:
            self.model_resnet = models.resnet50(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features


class VGG16Fc(BaseFeatureExtractor):
    def __init__(self,model_path=None, normalize=True):
        super(VGG16Fc, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_vgg = models.vgg16(pretrained=False)
                self.model_vgg.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_vgg = models.vgg16(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_vgg = self.model_vgg
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier"+str(i), model_vgg.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)

        self.__in_features = 4096

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features

class InceptionNet(BaseFeatureExtractor):
    """
    ** input image should be in range of [0, 1]**
    """
    def __init__(self,model_path=None, normalize=True):
        super(InceptionNet, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_inception = models.inception_v3(pretrained=False)
                self.model_inception.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_inception = models.inception_v3(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_inception = self.model_inception
        self.aux_logits = model_inception.aux_logits
        self.transform_input = model_inception.transform_input
        self.Conv2d_1a_3x3 = model_inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = model_inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = model_inception.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = model_inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = model_inception.Conv2d_4a_3x3
        self.Mixed_5b = model_inception.Mixed_5b
        self.Mixed_5c = model_inception.Mixed_5c
        self.Mixed_5d = model_inception.Mixed_5d
        self.Mixed_6a = model_inception.Mixed_6a
        self.Mixed_6b = model_inception.Mixed_6b
        self.Mixed_6c = model_inception.Mixed_6c
        self.Mixed_6d = model_inception.Mixed_6d
        self.Mixed_6e = model_inception.Mixed_6e
        self.Mixed_7a = model_inception.Mixed_7a 
        self.Mixed_7b = model_inception.Mixed_7b
        self.Mixed_7c = model_inception.Mixed_7c
        self.__in_features = model_inception.fc.in_features

    def forward(self, x):
        if self.transform_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = F.avg_pool2d(x, kernel_size=8)
        x = F.dropout(x, training=self.training)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features

class PnasNet(BaseFeatureExtractor):
    """
    ** input image should be in range of [0, 1]**
    """
    def __init__(self,model_path=None, normalize=True):
        super(PnasNet, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_pnas = pnasnet_model.pnasnet5large(pretrained=False)
                self.model_pnas.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_pnas = pnasnet_model.pnasnet5large(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_pnas = self.model_pnas
        self.conv_0 = model_pnas.conv_0
        self.cell_stem_0 = model_pnas.cell_stem_0
        self.cell_stem_1 = model_pnas.cell_stem_1
        self.cell_0 = model_pnas.cell_0
        self.cell_1 = model_pnas.cell_1
        self.cell_2 = model_pnas.cell_2
        self.cell_3 = model_pnas.cell_3
        self.cell_4 = model_pnas.cell_4
        self.cell_5 = model_pnas.cell_5
        self.cell_6 = model_pnas.cell_6
        self.cell_7 = model_pnas.cell_7
        self.cell_8 = model_pnas.cell_8
        self.cell_9 = model_pnas.cell_9
        self.cell_10 = model_pnas.cell_10
        self.cell_11 = model_pnas.cell_11
        self.relu = model_pnas.relu
        self.avg_pool = model_pnas.avg_pool
        self.dropout = model_pnas.dropout
        self.__in_features = model_pnas.last_linear.in_features

    def forward(self, x):
        x_conv_0 = self.conv_0(x)
        x_stem_0 = self.cell_stem_0(x_conv_0)
        x_stem_1 = self.cell_stem_1(x_conv_0, x_stem_0)
        x_cell_0 = self.cell_0(x_stem_0, x_stem_1)
        x_cell_1 = self.cell_1(x_stem_1, x_cell_0)
        x_cell_2 = self.cell_2(x_cell_0, x_cell_1)
        x_cell_3 = self.cell_3(x_cell_1, x_cell_2)
        x_cell_4 = self.cell_4(x_cell_2, x_cell_3)
        x_cell_5 = self.cell_5(x_cell_3, x_cell_4)
        x_cell_6 = self.cell_6(x_cell_4, x_cell_5)
        x_cell_7 = self.cell_7(x_cell_5, x_cell_6)
        x_cell_8 = self.cell_8(x_cell_6, x_cell_7)
        x_cell_9 = self.cell_9(x_cell_7, x_cell_8)
        x_cell_10 = self.cell_10(x_cell_8, x_cell_9)
        x_cell_11 = self.cell_11(x_cell_9, x_cell_10)
        x = self.relu(x_cell_11)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def output_num(self):
        return self.__in_features

class NasNet(BaseFeatureExtractor):
    """
    ** input image should be in range of [0, 1]**
    """
    def __init__(self,model_path=None, normalize=True):
        super(NasNet, self).__init__()
        if model_path:
            if os.path.exists(model_path):
                self.model_nas = nasnet_model.nasnetalarge(pretrained=False)
                self.model_nas.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        else:
            self.model_nas = nasnet_model.nasnetalarge(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_nas = self.model_nas
        self.stem_filters = model_nas.stem_filters
        self.penultimate_filters = model_nas.penultimate_filters
        self.filters_multiplier = model_nas.filters_multiplier

        filters = self.penultimate_filters // 24

        self.conv0 = model_nas.conv0
        self.conv0.add_module('conv', nn.Conv2d(in_channels=3, out_channels=self.stem_filters, kernel_size=3, padding=0, stride=2,
                                                bias=False))
        self.conv0.add_module('bn', nn.BatchNorm2d(self.stem_filters, eps=0.001, momentum=0.1, affine=True))
        self.cell_stem_0 = model_nas.cell_stem_0
        self.cell_stem_1 = model_nas.cell_stem_1
        self.cell_0 = model_nas.cell_0
        self.cell_1 = model_nas.cell_1
        self.cell_2 = model_nas.cell_2
        self.cell_3 = model_nas.cell_3
        self.cell_4 = model_nas.cell_4
        self.cell_5 = model_nas.cell_5
        self.reduction_cell_0 = model_nas.reduction_cell_0
        self.cell_6 = model_nas.cell_6
        self.cell_7 = model_nas.cell_7
        self.cell_8 = model_nas.cell_8
        self.cell_9 = model_nas.cell_9
        self.cell_10 = model_nas.cell_10
        self.cell_11 = model_nas.cell_11
        self.reduction_cell_1 = model_nas.reduction_cell_1
        self.cell_12 = model_nas.cell_12
        self.cell_13 = model_nas.cell_13
        self.cell_14 = model_nas.cell_14
        self.cell_15 = model_nas.cell_15
        self.cell_16 = model_nas.cell_16
        self.cell_17 = model_nas.cell_17
        self.relu = model_nas.relu
        self.avg_pool = model_nas.avg_pool
        self.dropout = model_nas.dropout
        self.__in_features = model_nas.last_linear.in_features

    def forward(self, x):
        x = self.conv0(x)
        x_stem_0 = self.cell_stem_0(x)
        x_stem_1 = self.cell_stem_1(x,x_stem_0)
        x_cell_0 = self.cell_0(x_stem_1, x_stem_0)
        x_cell_1 = self.cell_1(x_cell_0, x_stem_1)
        x_cell_2 = self.cell_2(x_cell_1, x_cell_0)
        x_cell_3 = self.cell_3(x_cell_2, x_cell_1)
        x_cell_4 = self.cell_4(x_cell_3, x_cell_2)
        x_cell_5 = self.cell_5(x_cell_4, x_cell_3)

        x_reduction_cell_0 = self.reduction_cell_0(x_cell_5, x_cell_4)

        x_cell_6 = self.cell_6(x_reduction_cell_0, x_cell_4)
        x_cell_7 = self.cell_7(x_cell_6, x_reduction_cell_0)
        x_cell_8 = self.cell_8(x_cell_7, x_cell_6)
        x_cell_9 = self.cell_9(x_cell_8, x_cell_7)
        x_cell_10 = self.cell_10(x_cell_9, x_cell_8)
        x_cell_11 = self.cell_11(x_cell_10, x_cell_9)

        x_reduction_cell_1 = self.reduction_cell_1(x_cell_11, x_cell_10)

        x_cell_12 = self.cell_12(x_reduction_cell_1, x_cell_10)
        x_cell_13 = self.cell_13(x_cell_12, x_reduction_cell_1)
        x_cell_14 = self.cell_14(x_cell_13, x_cell_12)
        x_cell_15 = self.cell_15(x_cell_14, x_cell_13)
        x_cell_16 = self.cell_16(x_cell_15, x_cell_14)
        x_cell_17 = self.cell_17(x_cell_16, x_cell_15)

        x = self.relu(x_cell_17)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.last_linear(x)
        return x

    def output_num(self):
        return self.__in_features

class GoogleNet(BaseFeatureExtractor):
    """
    ** input image should be in range of [0, 1]**
    """
    def __init__(self,model_path=None, normalize=True):
        super(GoogleNet, self).__init__()
        #如果本地有自己的resnet，则调用本地的resnet.pkl
        if model_path:
            if os.path.exists(model_path):
                self.model_googlenet = models.googlenet(pretrained=False)
                self.model_googlenet.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        #否则调用网上pretrain好的resnet，联网下载
        else:
            self.model_googlenet = models.googlenet(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_googlenet = self.model_googlenet
        self.aux_logits = model_googlenet.aux_logits
        self.transform_input = model_googlenet.transform_input
        self.conv1 = model_googlenet.conv1
        self.maxpool1 = model_googlenet.maxpool1
        self.conv2 = model_googlenet.conv2
        self.conv3 = model_googlenet.conv3
        self.maxpool2 = model_googlenet.maxpool2
        self.inception3a = model_googlenet.inception3a
        self.inception3b = model_googlenet.inception3b
        self.maxpool3 = model_googlenet.maxpool3
        self.inception4a = model_googlenet.inception4a
        self.inception4b = model_googlenet.inception4b
        self.inception4c = model_googlenet.inception4c
        self.inception4d = model_googlenet.inception4d
        self.inception4e = model_googlenet.inception4e
        self.maxpool4 = model_googlenet.maxpool4
        self.inception5a = model_googlenet.inception5a
        self.inception5b = model_googlenet.inception5b
        if self.aux_logits:
            self.aux1 = model_googlenet.aux1
            self.aux2 = model_googlenet.aux2

        self.avgpool = model_googlenet.avgpool
        self.dropout = model_googlenet.dropout
        self.__in_features = model_googlenet.fc.in_features

    def forward(self, x):
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56
        x = self.conv2(x)
        # N x 64 x 56 x 56
        x = self.conv3(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)

        # N x 192 x 28 x 28
        x = self.inception3a(x)
        # N x 256 x 28 x 28
        x = self.inception3b(x)
        # N x 480 x 28 x 28
        x = self.maxpool3(x)
        # N x 480 x 14 x 14
        x = self.inception4a(x)
        # N x 512 x 14 x 14
        aux_defined = self.training and self.aux_logits
        if aux_defined:
            aux1 = self.aux1(x)
        else:
            aux1 = None

        x = self.inception4b(x)
        # N x 512 x 14 x 14
        x = self.inception4c(x)
        # N x 512 x 14 x 14
        x = self.inception4d(x)
        # N x 528 x 14 x 14
        if aux_defined:
            aux2 = self.aux2(x)
        else:
            aux2 = None

        x = self.inception4e(x)
        # N x 832 x 14 x 14
        x = self.maxpool4(x)
        # N x 832 x 7 x 7
        x = self.inception5a(x)
        # N x 832 x 7 x 7
        x = self.inception5b(x)
        # N x 1024 x 7 x 7

        x = self.avgpool(x)
        # N x 1024 x 1 x 1
        x = torch.flatten(x, 1)
        # N x 1024
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        # N x 1000 (num_classes)
        return x

    def output_num(self):
        return self.__in_features

class CLS(nn.Module):
    """
    a two-layer MLP for classification
    """
    def __init__(self, in_dim, out_dim, bottle_neck_dim=256):
        super(CLS, self).__init__()
        self.bottleneck = nn.Sequential(
            nn.Linear(in_dim, bottle_neck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(bottle_neck_dim, bottle_neck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
            )
        self.fc = nn.Linear(bottle_neck_dim, out_dim)
        self.main = nn.Sequential(self.bottleneck, self.fc, nn.Softmax(dim=1))

    def forward(self, x):
        out = [x]
        #四层输入输出依次从CLS得到：layer1的输入，输出，layer2的输入，输出
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out


class AdversarialNetwork(nn.Module):
    """
    AdversarialNetwork with a gredient reverse layer.
    its ``forward`` function calls gredient reverse layer first, then applies ``self.main`` module.
    """
    def __init__(self, in_feature,out_dim):
        super(AdversarialNetwork, self).__init__()
        '''
        self.main = nn.Sequential(
            nn.Linear(in_feature, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, out_dim),
            nn.Sigmoid()
        )
        '''
        self.fc = nn.Sequential(
            nn.Linear(in_feature, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048,1024)
            #nn.ReLU(inplace=True),
            #nn.Dropout(0.5)
        )
        self.dis = nn.Sequential(
            nn.Linear(1024, out_dim),
            nn.Sigmoid()
        )
        self.main = nn.Sequential(self.fc, self.dis)
        self.grl = GradientReverseModule(lambda step: aToBSheduler(step, 0.0, 1.0, gamma=10, max_iter=10000))

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out
        '''
        x_ = self.grl(x)
        y_ = self.fc(x_)
        y = self.dis(y_)
        #y = self.main(x_)
        return y_,y
        '''
class WideResNet100Fc(BaseFeatureExtractor):
    """
    ** input image should be in range of [0, 1]**
    """
    def __init__(self,model_path=None, normalize=True):
        super(WideResNet100Fc, self).__init__()
        #如果本地有自己的resnet，则调用本地的resnet.pkl
        if model_path:
            if os.path.exists(model_path):
                self.model_resnet = models.wide_resnet101_2(pretrained=False)
                self.model_resnet.load_state_dict(torch.load(model_path))
            else:
                raise Exception('invalid model path!')
        #否则调用网上pretrain好的resnet，联网下载
        else:
            self.model_resnet = models.wide_resnet101_2(pretrained=True)

        if model_path or normalize:
            # pretrain model is used, use ImageNet normalization
            self.normalize = True
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        else:
            self.normalize = False

        model_resnet = self.model_resnet
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        if self.normalize:
            x = (x - self.mean) / self.std
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def output_num(self):
        return self.__in_features