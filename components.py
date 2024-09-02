import torch
import torch.nn as nn


class Multiscale_Feature_Extractor(nn.Module):
    def __init__(self, in_ch, out_ch, ch_step, res_layers):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ch_step = ch_step
        self.res_layers = res_layers

        self.conv0_0 = nn.Conv2d(in_channels=in_ch, out_channels=16, kernel_size=3, padding=1)
        self.conv0_1 = nn.Conv2d(in_channels=in_ch - 2 * ch_step, out_channels=16, kernel_size=3, padding=1)
        self.conv0_2 = nn.Conv2d(in_channels=in_ch - 4 * ch_step, out_channels=16, kernel_size=3, padding=1)
        self.conv0_3 = nn.Conv2d(in_channels=in_ch - 6 * ch_step, out_channels=16, kernel_size=3, padding=1)

        self.conv1_0 = nn.Conv2d(in_channels=16, out_channels=out_ch, kernel_size=3, padding=1)
        self.conv1_1 = nn.Conv2d(in_channels=16, out_channels=out_ch, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(in_channels=16, out_channels=out_ch, kernel_size=3, padding=1)
        self.conv1_3 = nn.Conv2d(in_channels=16, out_channels=out_ch, kernel_size=3, padding=1)

        self.res_net0 = nn.Sequential(*[BasicBlock(features=out_ch) for _ in range(res_layers)])
        self.res_net1 = nn.Sequential(*[BasicBlock(features=out_ch) for _ in range(res_layers)])
        self.res_net2 = nn.Sequential(*[BasicBlock(features=out_ch) for _ in range(res_layers)])
        self.res_net3 = nn.Sequential(*[BasicBlock(features=out_ch) for _ in range(res_layers)])

        self.relu = nn.ReLU()

    def forward(self, x, fea_idx=-1):
        res = []
        if fea_idx == 0 or fea_idx == -1:
            out_0 = self.res_net0(self.relu(self.conv1_0(self.relu(self.conv0_0(x)))))
            res.append(out_0)
        if fea_idx == 1 or fea_idx == -1:
            out_1 = self.res_net1(self.relu(self.conv1_1(self.relu(self.conv0_1(x[:, self.ch_step:-self.ch_step, :, :])))))
            res.append(out_1)
        if fea_idx == 2 or fea_idx == -1:
            out_2 = self.res_net2(self.relu(self.conv1_2(self.relu(self.conv0_2(x[:, 2 * self.ch_step:-2 * self.ch_step, :, :])))))
            res.append(out_2)
        if fea_idx == 3 or fea_idx == -1:
            out_3 = self.res_net3(self.relu(self.conv1_3(self.relu(self.conv0_3(x[:, 3 * self.ch_step:-3 * self.ch_step, :, :])))))
            res.append(out_3)

        return res
    

class Feature_Composition(nn.Module):
    def __init__(self, in_ch, out_ch, multi_scale, res_layers):
        super().__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.multi_scale = multi_scale
        self.res_layers = res_layers

        self.ca = CALayer_v2(in_channels=in_ch * multi_scale)
        self.conv0 = nn.Conv2d(in_channels=in_ch * multi_scale, out_channels=in_ch, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1)
        self.relu = nn.ReLU()

        self.res_net = nn.Sequential(*[BasicBlock(features=out_ch) for _ in range(res_layers)])

    def forward(self, x):
        if type(x) is list:
            x = torch.cat(x, dim=1)
        weight = self.ca(x)
        x = x * weight
        x = self.relu(self.conv1(self.relu(self.conv0(x))))
        y = self.res_net(x)
        return x + y


class BasicBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=True)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        return self.relu3(x + out)
    

class CALayer_v2(nn.Module):
    def __init__(self, in_channels):
        super(CALayer_v2, self).__init__()
        self.ca_block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, bias=True),
            nn.ReLU(),
            nn.Conv2d(in_channels * 2, in_channels, 3, padding=1, bias=True),
            nn.Sigmoid())

    def forward(self, x):
        weight = self.ca_block(x)
        return weight
    