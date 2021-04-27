import torch.nn as nn
import torch
class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
              nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.Conv2d(64, 192, kernel_size=3, padding=1),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
              nn.Conv2d(192, 384, kernel_size=3, padding=1),
              nn.ReLU(inplace=True),
              nn.Conv2d(384, 256, kernel_size=3, padding=1),
              nn.ReLU(inplace=True),
              nn.Conv2d(256, 256, kernel_size=3, padding=1),
              nn.ReLU(inplace=True),
              nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 4 * 4, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            #nn.Linear(4096, 4096),
            #nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )



    def forward(self, x):
        feat_conv1 = self.features[0](x)
        feat_conv1_relu = self.features[1](feat_conv1)
#         print(feat_conv1_relu1.size())
        feat_pool1 = self.features[2](feat_conv1_relu)
        feat_conv2 = self.features[3](feat_pool1)
        feat_conv2_relu = self.features[4](feat_conv2)
#         print(feat_conv2_relu2.size())
        feat_pool2 = self.features[5](feat_conv2_relu)
        feat_conv3 = self.features[6](feat_pool2)
        feat_conv3_relu = self.features[7](feat_conv3)
#         print(feat_conv3_relu3.size())
        feat_conv4 = self.features[8](feat_conv3_relu)
        feat_conv4_relu = self.features[9](feat_conv4)
#         print(feat_conv4_relu4.size())
        feat_conv5 = self.features[10](feat_conv4_relu)
        feat_conv5_relu = self.features[11](feat_conv5)
#         print(feat_conv5_relu5.size())
        feat_pool5 = self.features[12](feat_conv5_relu)
#         print(feat_pool5.size())
        
        x = feat_pool5.view(feat_pool5.size(0), -1)
#         y = self.classifier(x)
        y = self.classifier[0](x)
        y = self.classifier[1](y)
        y = self.classifier[2](y)
        y = self.classifier[3](y)
        y = self.classifier[4](y)
        
        return y
#         return y, {'feat_conv1_relu': feat_conv1_relu, 'feat_conv2_relu': feat_conv2_relu, 'feat_conv3_relu': feat_conv3_relu, 'feat_conv4_relu': feat_conv4_relu, 'feat_conv5_relu': feat_conv5_relu}

if __name__ == "__main__":
    net = AlexNet()
    input = torch.randn((1,3,32,32))
    output = net(input)
    print(output.size())
    for m in net.modules():
        if isinstance(m,nn.Conv2d):
            print(m.weight.size())