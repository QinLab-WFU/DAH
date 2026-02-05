import argparse
import os.path as osp


def get_config():
    parser = argparse.ArgumentParser(description=osp.basename(osp.dirname(__file__)))

    # common settings
    parser.add_argument("--backbone", type=str, default="resnet50", help="see network.py")
    parser.add_argument("--data-dir", type=str, default="../_datasets_zs", help="directory to dataset")
    parser.add_argument("--n-workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--n-epochs", type=int, default=100, help="number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for training")
    parser.add_argument("--optimizer", type=str, default="adam", help="sgd/rmsprop/adam/amsgrad/adamw")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")
    parser.add_argument("--wd", type=float, default=1e-4, help="weight decay")
    parser.add_argument("--device", type=str, default="cuda:0", help="device (accelerator) to use")
    parser.add_argument("--multi-thread", type=bool, default=True, help="use a separate thread for validation")

    # changed at runtime
    parser.add_argument("--dataset", type=str, default="cub", help="awa2/cub/sun")
    parser.add_argument("--n-classes", type=int, default=200, help="number of dataset classes")
    parser.add_argument("--topk", type=int, default=1000, help="mAP@topk")
    parser.add_argument("--save-dir", type=str, default="./output", help="directory to output results")
    parser.add_argument("--n-bits", type=int, default=16, help="length of hashing binary")

    # special settings
    parser.add_argument(
        "--samples_per_class",
        type=int,
        default=1,
        help="Number of samples in one class drawn before choosing the next class.",
    )

    parser.add_argument("--lambda1", type=float, default=0.001, help="weight of loss AR")
    parser.add_argument("--lambda2", type=float, default=1.0, help="weight of loss ACE")
    parser.add_argument("--lambda3", type=float, default=0.25, help="weight of loss SC")
    parser.add_argument("--lambda4", type=float, default=1.0, help="weight of loss TRI")

    args = parser.parse_args()

    # args.rename = True

    # code
    # args.backbone = "resnet101"
    # args.n_epochs = 400
    args.batch_size = 50
    args.optimizer = "sgd"
    args.lr = 1e-4
    # args.lambda4 = 0

    # mods
    # args.device = "cuda:1"

    return args
