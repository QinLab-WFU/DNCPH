import argparse
import os


def get_config():
    parser = argparse.ArgumentParser(description=os.path.basename(os.path.dirname(__file__)))

    # common settings
    parser.add_argument("--backbone", type=str, default="resnet50", help="see _network.py")
    parser.add_argument("--data-dir", type=str, default="../_datasets", help="directory to dataset")
    parser.add_argument("--n-workers", type=int, default=4, help="number of dataloader workers")
    parser.add_argument("--n-epochs", type=int, default=300, help="number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=128, help="batch size for training")
    parser.add_argument("--optimizer", type=str, default="adam", help="sgd/rmsprop/adam/amsgrad/adamw")
    parser.add_argument("--lr", type=float, default=0.045, help="learning rate")
    parser.add_argument("--wd", type=float, default=0, help="weight decay")
    parser.add_argument("--device", type=str, default="cuda:1", help="device (accelerator) to use")
    parser.add_argument("--multi-thread", type=bool, default=True, help="use a separate thread for validation")

    # will be changed at runtime
    parser.add_argument("--dataset", type=str, default="flickr", help="cifar/nuswide/flickr/coco")
    parser.add_argument("--n-classes", type=int, default=38, help="number of dataset classes")
    parser.add_argument("--topk", type=int, default=None, help="mAP@topk")
    parser.add_argument("--save-dir", type=str, default="./output", help="directory to output results")
    parser.add_argument("--n-bits", type=int, default=16, help="length of hashing binary")

    # special settings
    parser.add_argument("--lr2", type=float, default=5.0, help="learning rate of proxies")
    parser.add_argument("--eps", type=float, default=1.0, help="eps for Adam, default is 1e-8")
    # scale = 1/temperature
    parser.add_argument("--scaling_x", default=3, type=int, help="Scaling factor for the normalized embeddings")
    parser.add_argument("--scaling_p", default=3, type=int, help="Scaling factor for the normalized proxies")
    parser.add_argument("--smoothing_const", default=0.1, type=float, help="label smoothing constant")
    parser.add_argument("--log_first", default=True, type=bool, help="sum method for multi-label")

    args = parser.parse_args()


    # mods
    args.lr = 1e-5
    args.wd = 1e-4
    args.scaling_x = 1
    args.scaling_p = 2
    args.lr2 = 1e-2
    args.eps = 1e-8
    args.smoothing_const = 0
    args.log_first = False

    return args


