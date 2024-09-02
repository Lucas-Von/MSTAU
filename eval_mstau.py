import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader

from components import Multiscale_Feature_Extractor, Feature_Composition
from pred_model.mstau_model import MSTAU_Pred
from dat_reader import spk_provider


parser = argparse.ArgumentParser()

parser.add_argument('--spk_path_file', type=str, default='dataset\eval_spk_path.txt')
parser.add_argument('--half_win', type=int, default=6)
parser.add_argument('--input_len', type=int, default=10)
parser.add_argument('--batch', type=int, default=2)
parser.add_argument('--worker_num', type=int, default=2)
parser.add_argument('--device', type=str, default='cuda:0')

parser.add_argument('--load_extractor_checkpoint', type=str)
parser.add_argument('--load_predictor_checkpoint', type=str)
parser.add_argument('--load_compositor_checkpoint', type=str)
args = parser.parse_args()


class AverageMeter:
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def reset(self):
        self.__init__()


def main():
    eval_dataset = spk_provider(args.spk_path_file, half_win=args.half_win, total_len=args.input_len + 1, random_start=False)
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=args.batch,
        num_workers=args.worker_num,
        shuffle=False,
        drop_last=True,
        pin_memory=(args.device != "cpu")
    )

    extractor = Multiscale_Feature_Extractor(in_ch=13, out_ch=16, ch_step=1, res_layers=6)
    predictor = MSTAU_Pred(in_ch=16, mid_ch=64, out_ch=16, layer=4, shape=(256, 448), tau=5, K=4, train_aggregation_only=False)
    compositor = Feature_Composition(in_ch=16, out_ch=64, multi_scale=4, res_layers=6)
    criterion = nn.MSELoss()

    print("Loading extractor from", args.load_extractor_checkpoint)
    extractor.load_state_dict(torch.load(args.load_extractor_checkpoint)['state_dict'])
    extractor = extractor.to(args.device)

    print("Loading predictor from", args.load_predictor_checkpoint)
    predictor.load_state_dict(torch.load(args.load_predictor_checkpoint)['state_dict'])
    predictor = predictor.to(args.device)

    print("Loading compositor from", args.load_compositor_checkpoint)
    compositor.load_state_dict(torch.load(args.load_compositor_checkpoint)['state_dict'])
    compositor = compositor.to(args.device)

    extractor.eval()
    predictor.eval()
    compositor.eval()
    loss = AverageMeter()
    with torch.no_grad():
        for _, d in enumerate(eval_dataloader):
            spk_voxel = d.to(args.device)
            features = []
            with torch.no_grad():
                for start_idx in range(args.input_len + 1):
                    fea = extractor(spk_voxel[:, start_idx: start_idx + (2 * args.half_win + 1)], -1)
                    features.append(torch.stack(fea, dim=1))
            features = torch.stack(features, dim=1)
            predicted_fea = predictor(features[:, :-1])
            out_criterion = criterion(compositor(predicted_fea), compositor(features[:, -1]))
            loss.update(out_criterion.item())

    print( f"Average losses:\tLoss: {loss.avg:.10f}")


if __name__ == "__main__":
    main()
