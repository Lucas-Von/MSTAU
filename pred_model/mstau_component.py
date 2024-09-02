import torch
import torch.nn as nn
import math


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


class Aggregate(nn.Module):
    def __init__(self, in_ch, out_ch, K, cur_k, res_num, down_sample):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.K = K
        self.cur_k = cur_k
        self.res_num = res_num
        self.down_sample = down_sample

        self.conv_q = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_k = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.res_net = nn.Sequential(*[BasicBlock(features=out_ch) for _ in range(res_num)])
        self.softmax = nn.Softmax(dim=0)

        if down_sample > 0:
            ds_q = nn.Sequential(*[nn.Sequential(
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, stride=2),
                nn.ReLU()
            ) for _ in range(down_sample)])
            self.conv_q = nn.Sequential(
                self.conv_q,
                ds_q
            )
            ds_k = nn.Sequential(*[nn.Sequential(
                nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, padding=1, stride=2),
                nn.ReLU()
            ) for _ in range(down_sample)])
            self.conv_k = nn.Sequential(
                self.conv_k,
                ds_k
            )

    def forward(self, qk, v):
        assert qk.shape[0] == self.K
        Q = self.conv_q(qk[self.cur_k])
        K = [self.conv_k(qk[i]) for i in range(self.K)]
        V = torch.stack([self.conv_v(v[i]) for i in range(self.K)], dim=0)

        d = Q.shape[1] * Q.shape[2] * Q.shape[3]
        weights_list = []
        for i in range(self.K):
            weights_list.append((torch.matmul(Q, K[i].transpose(-2, -1))).sum(dim=(1, 2, 3)) / math.sqrt(d))
        weights_list = torch.stack(weights_list, dim=0)
        weights_list = torch.reshape(weights_list, (*weights_list.shape, 1, 1, 1))
        weights_list = self.softmax(weights_list)
        out = V * weights_list
        out = out.sum(dim=0)

        return out


class MSTAUCell(nn.Module):
    def __init__(self,
                 in_ch=16,
                 mid_ch=16,
                 shape=(256, 448),
                 tau=5,
                 K=4,
                 cur_k=-1):
        super(MSTAUCell, self).__init__()

        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.height = shape[0]
        self.width = shape[1]
        self.tau = tau
        self.K = K
        self.cur_k = cur_k
        self.d = mid_ch * shape[0] * shape[1]
        assert 0 <= self.cur_k < self.K

        self.conv_t = nn.Sequential(
            nn.Conv2d(in_ch, 3 * mid_ch, kernel_size=3, padding=1),
            nn.LayerNorm([3 * mid_ch, self.height, self.width])
        )
        self.conv_t_next = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.LayerNorm([mid_ch, self.height, self.width])
        )
        self.conv_s = nn.Sequential(
            nn.Conv2d(in_ch, 3 * mid_ch, kernel_size=3, padding=1),
            nn.LayerNorm([3 * mid_ch, self.height, self.width])
        )

        self.conv_s_next = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.LayerNorm([mid_ch, self.height, self.width])
        )
        self.softmax = nn.Softmax(dim=0)

        self.aggregate = Aggregate(in_ch=mid_ch, out_ch=mid_ch, K=K, cur_k=cur_k, res_num=8, down_sample=0)

    def forward(self, T_t, S_t, t_att, s_att):
        assert S_t.shape[:1] + S_t.shape[2:] == (self.K, self.in_ch, self.height, self.width)
        assert T_t.shape[1:] == (self.in_ch, self.height, self.width)
        assert s_att.shape[:2] + s_att.shape[3:] == (self.tau, self.K, self.in_ch, self.height, self.width)
        assert t_att.shape[:1] + t_att.shape[2:] == (self.tau, self.in_ch, self.height, self.width)

        t_trend_list = []
        s_next_list = []
        for k in range(self.K):
            s_t = S_t[k]
            s_next = self.conv_s_next(s_t)
            s_next_list.append(s_next)
            weights_list = []
            for t in range(self.tau):
                weights_list.append((s_att[t][k] * s_next).sum(dim=(1, 2, 3)) / math.sqrt(self.d))
            weights_list = torch.stack(weights_list, dim=0)
            weights_list = torch.reshape(weights_list, (*weights_list.shape, 1, 1, 1))
            weights_list = self.softmax(weights_list)
            t_trend = t_att * weights_list
            t_trend = t_trend.sum(dim=0)
            t_trend_list.append(t_trend)
            s_next_list = torch.stack(s_next_list, 0)
            t_trend_list = torch.stack(t_trend_list, 0)
            T_trend = self.aggregate(s_next_list, t_trend_list)            

        t_next = self.conv_t_next(T_t)
        t_att_gate = torch.sigmoid(t_next)
        T_fusion = T_t * t_att_gate + (1 - t_att_gate) * T_trend

        T_concat = self.conv_t(T_fusion)
        S_concat = self.conv_s(S_t[self.cur_k])
        t_g, t_t, t_s = torch.split(T_concat, self.mid_ch, dim=1)
        s_g, s_t, s_s = torch.split(S_concat, self.mid_ch, dim=1)
        T_gate = torch.sigmoid(t_g)
        S_gate = torch.sigmoid(s_g)
        T_new = T_gate * t_t + (1 - T_gate) * s_t
        S_new = S_gate * s_s + (1 - S_gate) * t_s
        return T_new, S_new
