import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
from losses import ln_loss, flow_loss
import torch.optim as optim
from restoration_task.thickness.utils import TrainSetLoader_thickness
from model import SwinUNet
from torch.utils.data import DataLoader

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")
# Training settings
parser = argparse.ArgumentParser(description="PyTorch")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--batchSize", type=int, default=200, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=10, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=2)
parser.add_argument("--threads", type=int, default=2, help="Number of threads for data loader to use, Default: 1")
parser.add_argument('--gamma', type=float, default=0.8, help='Learning Rate decay')


def train():
    opt = parser.parse_args()
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    model = SwinUNet(final_ch=5).cuda()

    # If you train your model from scratch, please delete this
    model.load_state_dict(torch.load("/data/Model/denoise_V10/HorusEye.pth"))
    for param in model.swin.swinViT.parameters():
        param.requires_grad = False
    for param in model.swin.encoder1.parameters():
        param.requires_grad = False
    for param in model.swin.encoder2.parameters():
        param.requires_grad = False
    for param in model.swin.encoder3.parameters():
        param.requires_grad = False
    for param in model.swin.encoder4.parameters():
        param.requires_grad = False

    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    print(model)

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    print("===> Training")
    for epoch in range(1, opt.nEpochs + 1):
        print(epoch)
        training_set = TrainSetLoader_thickness()
        training_loader = DataLoader(dataset=training_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                     shuffle=True)
        trainor(training_loader, optimizer, model, epoch)
        scheduler.step()


def trainor(training_loader, optimizer, model, epoch):
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    loss_epoch = 0
    for iteration, (destroyed, gt) in enumerate(training_loader):

        pre = model(destroyed)

        loss = ln_loss(pre, gt) + flow_loss(destroyed, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_epoch += loss

        print("===> Epoch[{}]: loss: {:.5f}  avg_loss: {:.5f}".format
              (epoch, loss, loss_epoch / (iteration % 200 + 1)))

        if (iteration + 1) % 200 == 0:
            loss_epoch = 0
            save_checkpoint(model, epoch, "/data/Model/HorusEye_V9")
            print("model has benn saved")


def save_checkpoint(model, epoch, path):
    model_out_path = os.path.join(path, "scratch_thickness.pth")
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    torch.multiprocessing.set_start_method("spawn")
    train()
