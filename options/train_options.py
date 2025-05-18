from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def __init__(self):
        super().__init__()

    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument('--optim', type=str, default='adam', help='optim to use [sgd, adam]')
        parser.add_argument('--loss_freq', type=int, default=100, help='frequency of showing loss on tensorboard')
        parser.add_argument('--save_epoch_freq', type=int, default=1,
                            help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_batch_freq', type=int, default=800,
                            help='frequency of saving checkpoints at the end of batches')
        parser.add_argument('--train_split', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--val_split', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--epoch', type=int, default=4, help='total epoches')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--pretrained_model', type=str, default="/root/autodl-tmp/checkpoints/experiment_name/final_lip_model_epoch_1.pth", help='model will fine tune on it if fine-tune is True')
        parser.add_argument('--fine-tune', type=bool, default=True)
        self.isTrain = True

        return parser