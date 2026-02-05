import os
import pickle

import torch
import torchvision
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from tqdm import tqdm

from _data_zs import ImageDataset, init_dataset, BalancedSampler


def build_trans(img_size=224):
    data_transforms = T.Compose(
        [
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return data_transforms


def calc_acc(logits, labels):
    predicted_labels = torch.zeros_like(labels)
    for i, k in enumerate(torch.count_nonzero(labels, dim=1)):
        # for i, k in enumerate(torch.sum(labels, dim=1)):
        predicted_labels[i, torch.topk(logits[i], k).indices] = 1
    acc = accuracy_score(labels.cpu(), predicted_labels.cpu())
    return acc


class FeatureDataset(Dataset):
    def __init__(self, feas, labs):
        self.feas, self.labs = feas, labs

    def __len__(self):
        return self.labs.shape[0]

    def __getitem__(self, idx):
        return self.feas[idx], self.labs[idx], idx

    def get_all_labels(self):
        return self.labs


def extract_features(net, dataloader, out_idx=None, usage=""):
    device = next(net.parameters()).device
    net.eval()

    feas = []
    for batch in tqdm(dataloader, desc=f"Extracting {usage} features".strip()):
        with torch.no_grad():
            out = net(batch[0].to(device))
        if out_idx is None:
            feas.append(out.cpu())
        else:
            feas.append(out[out_idx].cpu())
    feas = torch.cat(feas)
    return feas


def prepare_loaders(args):
    paper_trans = build_trans(224 if args.backbone == "resnet50" else 448)
    data = init_dataset(args.dataset, args.data_dir)
    kwargs = {"batch_size": args.batch_size, "num_workers": args.n_workers}
    train_loader = DataLoader(ImageDataset(data.train, paper_trans), generator=torch.Generator(), **kwargs)
    query_loader = DataLoader(ImageDataset(data.query, paper_trans), generator=torch.Generator(), **kwargs)
    dbase_loader = DataLoader(ImageDataset(data.dbase, paper_trans), generator=torch.Generator(), **kwargs)

    # build the backbone
    weights = torchvision.models.get_model_weights(args.backbone)["IMAGENET1K_V1"]
    resnet = torchvision.models.__dict__[args.backbone](weights=weights)
    args.feat_dim = resnet.fc.in_features
    backbone = nn.Sequential(*list(resnet.children())[:-2]).eval()
    backbone.to(args.device)

    cache_path = f"/tmp/TransZero_{args.backbone}_{args.dataset}_cache.pkl"
    if os.path.exists(cache_path):
        # use cache
        print("Using cached features...")
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        tF, qF, rF = data["tF"], data["qF"], data["rF"]
    else:
        tF = extract_features(backbone, train_loader, usage="train")
        qF = extract_features(backbone, query_loader, usage="query")
        rF = extract_features(backbone, dbase_loader, usage="dbase")
        # save cache
        save_obj = {
            "tF": tF,
            "qF": qF,
            "rF": rF,
        }
        with open(cache_path, "ab") as f:
            pickle.dump(save_obj, f)

    train_set = FeatureDataset(tF, train_loader.dataset.get_all_labels())
    query_set = FeatureDataset(qF, query_loader.dataset.get_all_labels())
    dbase_set = FeatureDataset(rF, dbase_loader.dataset.get_all_labels())

    query_loader2 = DataLoader(query_set, generator=torch.Generator(), **kwargs)
    dbase_loader2 = DataLoader(dbase_set, generator=torch.Generator(), **kwargs)

    batch_size = kwargs.pop("batch_size")
    sampler = BalancedSampler(train_set, batch_size=batch_size, samples_per_class=args.samples_per_class)
    train_loader2 = DataLoader(train_set, batch_sampler=sampler, **kwargs)

    return train_loader2, query_loader2, dbase_loader2


if __name__ == "__main__":
    pass
