from argparse import Namespace

import torch
import torch.nn.functional as F
from torch import nn

from DCHMT_CM.network import LinearHash
from _data_zs import get_w2vs
from transformer import Transformer


def build_model(args: Namespace):
    if args.backbone in ["resnet50", "resnet101"]:
        w2vs = get_w2vs(args.dataset, args.data_dir)  # normalized
        net = TransZero(args, w2vs).to(args.device)
        return net, 1

    raise NotImplementedError(f"Not support: {args.backbone}")


class TransZero(nn.Module):
    def __init__(self, args: Namespace, w2vs):
        super().__init__()

        d_model = 128  # dim of transformer

        # GloVe features for attributes name
        # a set of learnable semantic embeddings vA as queries
        self.V = nn.Parameter(w2vs)

        # mapping
        self.W_1 = nn.Parameter(nn.init.normal_(torch.empty(300, d_model)))

        # mapping for embeddings
        self.hash_layer = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, args.n_bits))
        # self.hash_layer = nn.Sequential(
        #     nn.LayerNorm(d_model), nn.Linear(d_model, d_model), LinearHash(d_model, args.n_bits)
        # )

        self.transformer = Transformer(
            ec_layer=1,
            dc_layer=1,
            dim_com=d_model,
            dim_feedforward=512,
            dropout=0.3,
            SAtt=True,
            heads=1,
        )

    def forward(self, x):
        # visual: B x C x H x W  -> B x C x H*W
        x = x.flatten(start_dim=2)
        x = F.normalize(x, dim=1)

        # locality-augmented visual features: maybe F
        trans_out = self.transformer(x, self.V)  # B x A x K

        # embedding to semantic space: maybe Eq. 13
        att_scores = torch.einsum("iv,vf,bif->bi", self.V, self.W_1, trans_out)

        embeddings = F.normalize(self.hash_layer(trans_out[:, 0, :]))
        # embeddings = self.hash_layer(trans_out[:, 0, :])

        return att_scores, embeddings


if __name__ == "__main__":
    args = Namespace(backbone="resnet50", dataset="cub", data_dir="../_datasets_zs", n_bits=16, device="cuda:0")
    net, _ = build_model(args)

    batch = torch.randn(2, 2048, 7, 7).to(args.device)
    out = net(batch)
    for x in out:
        print(x.shape)
