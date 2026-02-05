import json
import os
import time

import torch
from loguru import logger


from _data_zs import get_class_num, get_topk
from _utils import (
    AverageMeter,
    build_optimizer,
    calc_map_eval,
    calc_learnable_params,
    EarlyStopping,
    init,
    print_in_md,
    rename_output,
    save_checkpoint,
    seed_everything,
    validate_smart,
    predict,
)
from config import get_config
from loss import TransZeroLoss
from network import build_model
from utils import calc_acc, prepare_loaders


def train_epoch(args, dataloader, net, criterion, optimizer, epoch):
    tic = time.time()

    stat_meters = {}
    for x in ["AR", "ACE", "SC", "NCA", "loss", "mAP"]:
        stat_meters[x] = AverageMeter()

    net.train()
    for features, labels, indices in dataloader:
        features, labels = features.to(args.device), labels.to(args.device)

        att_scores, embeddings = net(features)

        loss1, loss2, loss3, loss4 = criterion(att_scores, embeddings, labels)
        stat_meters["AR"].update(loss1.item())
        stat_meters["ACE"].update(loss2.item())
        stat_meters["SC"].update(loss3.item())
        stat_meters["NCA"].update(loss4.item())

        loss = args.lambda1 * loss1 + args.lambda2 * loss2 + args.lambda3 * loss3 + args.lambda4 * loss4
        stat_meters["loss"].update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # to check overfitting
        map_v = calc_map_eval(embeddings.detach().sign(), labels)
        stat_meters["mAP"].update(map_v)

        torch.cuda.empty_cache()

    toc = time.time()
    sm_str = ""
    for x in stat_meters.keys():
        sm_str += f"[{x}:{stat_meters[x].avg:.4f}]"
    logger.info(
        f"[Training][dataset:{args.dataset}][bits:{args.n_bits}][epoch:{epoch}/{args.n_epochs - 1}][time:{(toc - tic):.3f}]{sm_str}"
    )


def train_init(args):
    # setup net
    net, out_idx = build_model(args)

    # setup criterion
    criterion = TransZeroLoss(args).to(args.device)

    logger.info(f"Number of net's params: {calc_learnable_params(net, criterion)}")
    # logger.info(f"Number of net's params: {calc_learnable_params(net)}")

    # setup optimizer
    to_optim = [
        {"params": net.parameters(), "lr": args.lr, "weight_decay": args.wd},
        {"params": criterion.parameters(), "lr": 100 * args.lr},
    ]
    if args.optimizer == "sgd":
        optimizer = build_optimizer(args.optimizer, to_optim, momentum=0.9)
    else:
        optimizer = build_optimizer(args.optimizer, to_optim)

    return net, out_idx, criterion, optimizer


def train(args, train_loader, query_loader, dbase_loader):
    net, out_idx, criterion, optimizer = train_init(args)

    early_stopping = EarlyStopping()

    for epoch in range(args.n_epochs):
        train_epoch(args, train_loader, net, criterion, optimizer, epoch)

        # we monitor mAP@topk validation accuracy every 5 epochs
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.n_epochs:
            # TODO: need a better way!
            qS, qL = predict(net, query_loader, out_idx=0, use_sign=False)
            qP = qS @ criterion.atts.T
            bias_mat = torch.ones_like(qP)
            bias_mat[:, args.seen_indices] = -1
            qP = qP + bias_mat
            aac_v = calc_acc(qP, qL)
            print(f"acc: {aac_v:.3f}")

            early_stop = validate_smart(
                args,
                query_loader,
                dbase_loader,
                early_stopping,
                epoch,
                model=net,
                out_idx=out_idx,
                multi_thread=args.multi_thread,
            )
            if early_stop:
                break

    if early_stopping.counter == early_stopping.patience:
        logger.info(
            f"Without improvement, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )
    else:
        logger.info(
            f"Reach epoch limit, will save & exit, best mAP: {early_stopping.best_map:.3f}, best epoch: {early_stopping.best_epoch}"
        )

    save_checkpoint(args, early_stopping.best_checkpoint)

    return early_stopping.best_epoch, early_stopping.best_map


def main():
    init()

    args = get_config()

    if "rename" in args and args.rename:
        rename_output(args)

    dummy_logger_id = None
    rst = []
    for dataset in ["awa2", "cub", "sun"]:
        # for dataset in ["sun"]:
        print(f"Processing dataset: {dataset}")
        args.dataset = dataset
        args.n_classes = get_class_num(dataset)
        args.topk = get_topk(dataset)

        # train_loader, query_loader, dbase_loader = build_loaders(
        #     dataset, args.data_dir, batch_size=args.batch_size, num_workers=args.n_workers
        # )
        # use fix data augmentation for train_loader
        train_loader, query_loader, dbase_loader = prepare_loaders(args)

        args.n_samples = len(train_loader.dataset)
        args.seen_indices = train_loader.dataset.get_all_labels().sum(dim=0).nonzero(as_tuple=True)[0]
        args.unseen_idxes = query_loader.dataset.get_all_labels().sum(dim=0).nonzero(as_tuple=True)[0]

        for hash_bit in [16, 32, 64, 128]:
            # for hash_bit in [16, 128]:
            print(f"Processing hash-bit: {hash_bit}")
            seed_everything()
            args.n_bits = hash_bit

            args.save_dir = f"./output/{args.backbone}/{dataset}/{hash_bit}"
            os.makedirs(args.save_dir, exist_ok=True)
            if any(x.endswith(".pth") for x in os.listdir(args.save_dir)):
                print(f"*.pth exists in {args.save_dir}, will pass")
                continue

            if dummy_logger_id is not None:
                logger.remove(dummy_logger_id)
            dummy_logger_id = logger.add(f"{args.save_dir}/train.log", mode="w", level="INFO")

            with open(f"{args.save_dir}/config.json", "w") as f:
                json.dump(
                    vars(args),
                    f,
                    indent=4,
                    sort_keys=True,
                    default=lambda o: o if type(o) in [bool, int, float, str] else str(type(o)),
                )

            best_epoch, best_map = train(args, train_loader, query_loader, dbase_loader)
            rst.append({"dataset": dataset, "hash_bit": hash_bit, "best_epoch": best_epoch, "best_map": best_map})

    print_in_md(rst)


if __name__ == "__main__":
    main()
