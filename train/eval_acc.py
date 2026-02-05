from argparse import Namespace

import torch

from _data_zs import get_atts
from _utils import predict
from network import build_model
from train import prepare_loaders
from utils import calc_acc

if __name__ == "__main__":

    args = Namespace(
        backbone="resnet101",
        dataset="sun",
        data_dir="../_datasets_zs",
        n_bits=128,
        batch_size=50,
        n_workers=4,
        samples_per_class=1,
        device="cuda:0",
    )

    net, _ = build_model(args, False)

    checkpoint = torch.load(f"./output/{args.backbone}/{args.dataset}/{args.n_bits}/e79_0.262.pth", map_location="cpu")
    msg = net.load_state_dict(checkpoint["model"])
    print(f"model loaded: {msg}")

    atts = get_atts(args.dataset, args.data_dir).to(args.device)
    if args.dataset == "awa2":
        # threshold at zero attribute with negative value
        atts[atts < 0] = 0

    _, query_loader, dbase_loader = prepare_loaders(args)

    qS, qL = predict(net, query_loader, out_idx=1, use_sign=False)
    qP = qS @ atts.T

    print(f"qL: {qL.argmax(1).sort()}")

    aac_v = calc_acc(qP, qL)
    print(f"acc: {aac_v:.3f}")
