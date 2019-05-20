import torch.nn as nn

from ptsemseg.models.utils import unetConv2, unetUp


class unet(nn.Module):
    def __init__(
        self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True
    ):
        super(unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv2d(filters[0], n_classes, 1)

    def forward(self, inputs):
        print(inputs.size())
        conv1 = self.conv1(inputs)
        print(conv1.size())

        maxpool1 = self.maxpool1(conv1)
        print(maxpool1.size())

        conv2 = self.conv2(maxpool1)
        print(conv2.size())

        maxpool2 = self.maxpool2(conv2)
        print(maxpool2.size())

        conv3 = self.conv3(maxpool2)
        print(conv3.size())

        maxpool3 = self.maxpool3(conv3)
        print(maxpool3.size())

        conv4 = self.conv4(maxpool3)
        print("conv4", conv4.size())

        maxpool4 = self.maxpool4(conv4)
        print("mp4", maxpool4.size())

        center = self.center(maxpool4)
        print("center", center.size())

        print(conv4.size(), center.size())
        up4 = self.up_concat4(conv4, center)
        print(conv3.size(), up4.size())
        up3 = self.up_concat3(conv3, up4)
        print(conv2.size(), up3.size())
        up2 = self.up_concat2(conv2, up3)
        print(conv1.size(), up2.size())
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final
