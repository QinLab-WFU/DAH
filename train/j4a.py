import torch
import torchvision
from sklearn.metrics import accuracy_score
from torch import nn
from torchvision import transforms as T


def build_trans():
    data_transforms = T.Compose(
        [
            T.Resize(448),
            T.CenterCrop(448),
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


def compute_per_class_acc_v2(logits, labels, n_classes):
    predicted_labels = torch.argmax(logits, 1)
    acc_per_class = torch.FloatTensor(n_classes).fill_(0)
    for i in range(n_classes):
        mask = labels == i
        acc_per_class[i] = torch.sum(labels[mask] == predicted_labels[mask]).float() / torch.sum(mask).float()
    return acc_per_class.mean().item()


def compute_per_class_acc(test_label, predicted_label, nclass):
    acc_per_class = torch.FloatTensor(nclass).fill_(0)
    for i in range(nclass):
        idx = test_label == i
        acc_per_class[i] = torch.sum(test_label[idx] == predicted_label[idx]).float() / torch.sum(idx).float()
    return acc_per_class.mean().item()


def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
    per_class_accuracies = torch.zeros(target_classes.size()[0]).float()
    predicted_label = predicted_label
    for i in range(target_classes.size()[0]):
        is_class = test_label == target_classes[i]
        per_class_accuracies[i] = torch.div(
            (predicted_label[is_class] == test_label[is_class]).sum().float(), is_class.sum().float()
        )
    return per_class_accuracies.mean().item()


def prepare_loaders(args):
    weights = torchvision.models.get_model_weights(args.backbone)["IMAGENET1K_V1"]
    resnet = torchvision.models.__dict__[args.backbone](weights=weights)
    backbone = nn.Sequential(*list(resnet.children())[:-2])
    args.feat_dim = resnet.fc.in_features

    paper_trans = build_trans()

    data = init_dataset(args.dataset, args.data_dir)

    train_set = ImageDataset(data.train, paper_trans)
    sampler = BalancedSampler(train_set, batch_size=args.batch_size, samples_per_class=args.samples_per_class)
    train_loader = DataLoader(train_set, batch_sampler=sampler, num_workers=args.n_workers)

    kwargs = {"batch_size": args.batch_size, "num_workers": args.n_workers}
    # generator=torch.Generator(): to keep torch.get_rng_state() unchanged!
    # https://discuss.pytorch.org/t/does-a-dataloader-change-random-state-even-when-shuffle-argument-is-false/92569/4
    query_loader = DataLoader(ImageDataset(data.query, paper_trans), generator=torch.Generator(), **kwargs)
    dbase_loader = DataLoader(ImageDataset(data.dbase, paper_trans), generator=torch.Generator(), **kwargs)

    return train_loader, query_loader, dbase_loader

if __name__ == "__main__":
    import torch.nn.functional as F

    C = 20

    logits = torch.randn(100, C)
    # labels = torch.randint(0, 10, (100,))
    labels = torch.randint(0, 10, (100,)) + 10
    onehot = F.one_hot(labels, C)

    print(calc_acc(logits, onehot))
    # print(compute_per_class_acc(labels, torch.argmax(logits, 1), 10))
    # print(compute_per_class_acc_v2(logits, labels, 10))
    print(compute_per_class_acc_gzsl(labels, torch.argmax(logits, 1), torch.arange(10) + 10))
