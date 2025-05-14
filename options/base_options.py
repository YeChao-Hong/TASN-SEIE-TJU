import os
import argparse
import torch


class BaseOptions:
    def __init__(self):
        self.initialized = False
        self.isTrain = False  # 初始化 isTrain 属性

    def initialize(self, parser):
        parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14", help="see models/__init__.py")
        parser.add_argument("--fix_backbone", default=False)
        parser.add_argument("--fix_encoder", default=True)
        # parser.add_argument("--real_list_path", default="./datasets/AVLips/0_real")  #注意这个是val路径，训练要改
        # parser.add_argument("--fake_list_path", default="./datasets/AVLips/1_fake")  #注意这个是val路径，训练要改
        parser.add_argument("--real_list_path", default="/root/autodl-tmp/val_036/0_real")
        parser.add_argument("--fake_list_path", default="/root/autodl-tmp/val_036/1_fake")
        parser.add_argument("--data_label", default="train", help="label to decide whether train or validation dataset")
        parser.add_argument("--batch_size", type=int, default=5, help="input batch size")
        parser.add_argument("--gpu_ids", type=str, default="0", help="gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU")
        parser.add_argument("--name", type=str, default="experiment_name", help="name of the experiment. It decides where to store samples and models")
        parser.add_argument("--num_threads", default=0, type=int, help="# threads for loading data")
        parser.add_argument("--checkpoints_dir", type=str, default="/root/autodl-tmp/checkpoints", help="models are saved here")
        parser.add_argument("--serial_batches", action="store_true", help="if true, takes images in order to make batches, otherwise takes them randomly")
        parser.add_argument("--suffix", type=str, default="", help="suffix for experiment name")
        parser.add_argument("--classes", type=str, default="", help="classes information")
        parser.add_argument("--rz_interp", type=str, default="", help="rz_interp information")
        parser.add_argument("--blur_sig", type=str, default="", help="blur_sig information")
        parser.add_argument("--jpg_method", type=str, default="", help="jpg_method information")
        parser.add_argument("--jpg_qual", type=str, default="", help="jpg_qual information")
        parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
        parser.add_argument('--class_bal',default=False, action='store_true', help='Whether to use class balancing')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ""
        message += "----------------- Options ---------------\n"
        for k, v in sorted(vars(opt).items()):
            comment = ""
            default = self.parser.get_default(k)
            if v != default:
                comment = "\t[default: %s]" % str(default)
            message += "{:>25}: {:<30}{}\n".format(str(k), str(v), comment)
        message += "----------------- End -------------------"
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        os.makedirs(expr_dir, exist_ok=True)
        # util.mkdirs(expr_dir)  # 如果不需要使用 util.mkdirs，可以删除这行代码
        file_name = os.path.join(expr_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write(message)
            opt_file.write("\n")

    def parse(self, print_options=True):
        opt = self.gather_options()
        opt.isTrain = self.isTrain  # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ("_" + opt.suffix.format(**vars(opt))) if opt.suffix != "" else ""
            opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt)

            # 根据 opt.gpu_ids 设置 GPU 设备
            gpu_ids = [int(id) for id in opt.gpu_ids.split(',') if id.strip()]
            num_gpus = torch.cuda.device_count()
            if gpu_ids and gpu_ids[0] != -1 and torch.cuda.is_available():
                if all(0 <= id < num_gpus for id in gpu_ids):
                    device = torch.device(f"cuda:{gpu_ids[0]}")
                    torch.cuda.set_device(device)
                else:
                    if num_gpus > 0:
                        print(f"Invalid GPU device IDs: {gpu_ids}. Using default GPU 0.")
                        device = torch.device("cuda:0")
                        torch.cuda.set_device(device)
                    else:
                        print(f"Invalid GPU device IDs: {gpu_ids}. No available GPUs, using CPU.")
                        device = torch.device("cpu")
            else:
                device = torch.device("cpu")

        # additional
        # opt.classes = opt.classes.split(',')
        try:
            opt.rz_interp = opt.rz_interp.split(",") if opt.rz_interp else []
            opt.blur_sig = [float(s) for s in opt.blur_sig.split(",")] if opt.blur_sig else []
            opt.jpg_method = opt.jpg_method.split(",") if opt.jpg_method else []
            opt.jpg_qual = [int(s) for s in opt.jpg_qual.split(",")] if opt.jpg_qual else []
            if len(opt.jpg_qual) == 2:
                opt.jpg_qual = list(range(opt.jpg_qual[0], opt.jpg_qual[1] + 1))
            elif len(opt.jpg_qual) > 2:
                raise ValueError("Shouldn't have more than 2 values for --jpg_qual.")
        except ValueError as e:
            print(f"Error processing parameters: {e}")
            raise

        self.opt = opt
        return self.opt


if __name__ == "__main__":
    options = BaseOptions()
    opt = options.parse()
    print(opt)
