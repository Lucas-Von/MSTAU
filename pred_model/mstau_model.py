import torch
import torch.nn as nn

from pred_model.mstau_component import MSTAUCell


class MSTAU_Pred(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, layer, shape, tau, K, train_aggregation_only=False):
        super().__init__()

        self.in_ch = in_ch
        self.mid_ch = mid_ch
        self.out_ch = out_ch
        self.layer = layer
        self.in_shape = shape
        self.mid_shape = (shape[0] // 8, shape[1] // 8)
        self.tau = tau
        self.K = K
        self.train_aggregation_only = train_aggregation_only

        self.enc_list = nn.ModuleList()
        self.dec_list = nn.ModuleList()
        for _ in range(K):
            enc = nn.Sequential(
                nn.Conv2d(in_channels=in_ch, out_channels=mid_ch, kernel_size=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=mid_ch, out_channels=mid_ch, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=mid_ch, out_channels=mid_ch, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=mid_ch, out_channels=mid_ch, kernel_size=3, padding=1, stride=2),
                nn.LeakyReLU(0.2)
            )
            dec = nn.Sequential(
                nn.ConvTranspose2d(in_channels=mid_ch, out_channels=mid_ch, kernel_size=3, padding=1, stride=2,
                                   output_padding=1),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(in_channels=mid_ch, out_channels=mid_ch, kernel_size=3, padding=1, stride=2,
                                   output_padding=1),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(in_channels=mid_ch, out_channels=mid_ch, kernel_size=3, padding=1, stride=2,
                                   output_padding=1),
                nn.LeakyReLU(0.2),
                nn.ConvTranspose2d(in_channels=mid_ch, out_channels=out_ch, kernel_size=1),
                nn.LeakyReLU(0.2)
            )
            self.enc_list.append(enc)
            self.dec_list.append(dec)

        self.cell_list = nn.ModuleList()
        for _ in range(layer):
            cell_layer_i = nn.ModuleList()
            for k in range(K):
                cell = MSTAUCell(in_ch=mid_ch, mid_ch=mid_ch, shape=self.mid_shape, tau=tau, K=K, cur_k=k)
                cell_layer_i.append(cell)
            self.cell_list.append(cell_layer_i)

        if train_aggregation_only:
            self.set_required_grad()


    def set_required_grad(self):
        for param in self.parameters():
            param.requires_grad = False

        for l in range(self.layer):
            for k in range(self.K):
                cell = self.cell_list[l][k]
                for param in cell.aggregate.parameters():
                    param.requires_grad = True


    def encode(self, input_fea):
        assert input_fea.shape[1] == self.K
        output_fea = []

        for k in range(self.K):
            input_fea_k = input_fea[:, k]
            output_fea_k = self.enc_list[k](input_fea_k)
            output_fea.append(output_fea_k)
        output_fea = torch.stack(output_fea, 1)

        return output_fea


    def decode(self, input_fea):
        assert input_fea.shape[1] == self.K
        output_fea = []

        for k in range(self.K):
            input_fea_k = input_fea[:, k]
            output_fea_k = self.dec_list[k](input_fea_k)
            output_fea.append(output_fea_k)
        output_fea = torch.stack(output_fea, 1)

        return output_fea


    def forward(self, x):
        # x: B*T*K*C*H*W
        assert len(x.shape) == 6
        B, T, K, C, H, W = x.shape
        assert K == self.K
        assert C == self.in_ch
        assert (H, W) == self.in_shape
        device = x.device

        T_pre = [
            [torch.zeros([K, B, self.mid_ch, self.mid_shape[0], self.mid_shape[1]]).to(device) for _ in
             range(self.tau)] for _ in range(self.layer)]  # L*t*K*B*C*H*W
        S_pre = [
            [torch.zeros([K, B, self.mid_ch, self.mid_shape[0], self.mid_shape[1]]).to(device) for _ in
             range(self.tau)] for _ in range(self.layer)]  # L*t*K*B*C*H*W
        s_t = None
        for t in range(T):
            if self.train_aggregation_only:
                with torch.no_grad():
                    s_t = self.encode(x[:, t]).permute(1, 0, 2, 3, 4).contiguous()
            else:
                s_t = self.encode(x[:, t]).permute(1, 0, 2, 3, 4).contiguous()
            for l in range(self.layer):
                t_t = T_pre[l][-1]
                t_att = torch.stack(T_pre[l], dim=0)
                s_att = torch.stack(S_pre[l], dim=0)
                t_out = []
                s_out = []
                for k in range(K):
                    cell_k = self.cell_list[l][k]
                    t_out_k, s_out_k= cell_k(t_t[k], s_t, t_att[:, k], s_att)
                    t_out.append(t_out_k)
                    s_out.append(s_out_k)
                t_out = torch.stack(t_out, dim=0)
                s_out = torch.stack(s_out, dim=0)
                T_pre[l].pop(0)
                S_pre[l].pop(0)
                T_pre[l].append(t_out)
                S_pre[l].append(s_out)
                s_t = s_out

        pred_x = self.decode(s_t.permute(1, 0, 2, 3, 4).contiguous())
        return pred_x
