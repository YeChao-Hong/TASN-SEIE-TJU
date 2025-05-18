import os
from torch.optim import lr_scheduler
import tensorflow as tf
from validate import validate
from data import create_dataloader
from trainer.trainer import Trainer
from options.train_options import TrainOptions
from tqdm import tqdm

# 设置 TensorFlow 日志级别，减少不必要的日志输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.data_label = "val"
    # val_opt.real_list_path = "./datasets/val/0_real"
    # val_opt.fake_list_path = "./datasets/val/1_fake"
    val_opt.real_list_path = "/root/autodl-tmp/val_036/0_real"
    val_opt.fake_list_path = "/root/autodl-tmp/val_036/1_fake"
    # val_opt.real_list_path = "/root/autodl-tmp/val_final/0_real"
    # val_opt.fake_list_path = "/root/autodl-tmp/val_final/1_fake"
    return val_opt

if __name__ == "__main__":
    opt = TrainOptions().parse()
    val_opt = get_val_opt()
    model = Trainer(opt)

    data_loader = create_dataloader(opt)
    val_loader = create_dataloader(val_opt)

    print("Length of data loader: %d" % (len(data_loader)))
    print("Length of val  loader: %d" % (len(val_loader)))

    # scheduler = lr_scheduler.ReduceLROnPlateau(model.optimizer, mode='min', factor=0.1, patience=5)
    scheduler = lr_scheduler.ReduceLROnPlateau(model.optimizer, mode='max', factor=0.1, patience=5)
    for epoch in range(opt.epoch):
        model.train()
        print(f"Epoch: {epoch + model.step_bias}")

        running_loss = 0.0

        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + model.step_bias} Training", unit="batch", position=0)
        for i, (img, crops, label) in enumerate(progress_bar):
            model.total_steps += 1

            model.set_input((img, crops, label))
            model.forward()
            loss = model.get_loss()
            model.optimize_parameters()
            running_loss += loss
            avg_loss = running_loss / (i + 1)

            # 更新进度条显示的 loss
            progress_bar.set_postfix({"Loss": avg_loss})

            # if epoch % opt.save_epoch_freq == 0 and i % opt.save_batch_freq == 0 and i!=0:
            #     print("saving the model at the end of batch %d" % (i+epoch*2953))
            #     model.save_networks("test1_model_bacth_%s.pth" % (i+epoch*2953))

        if epoch % opt.save_epoch_freq == 0:
            print("saving the model at the end of epoch %d" % (epoch + model.step_bias))
            model.save_networks("final_lip_model_epoch_%s.pth" % (epoch + model.step_bias))

        model.eval()
        ap, fpr, fnr, acc = validate(model.model, val_loader, opt.gpu_ids)
        print(
            "(Val @ epoch {}) acc: {} ap: {} fpr: {} fnr: {}".format(
                epoch + model.step_bias, acc, ap, fpr, fnr
            )
        )
        scheduler.step(acc)