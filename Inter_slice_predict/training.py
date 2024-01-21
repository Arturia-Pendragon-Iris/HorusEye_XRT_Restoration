import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from models import UNet_2D
from utils import TrainSetLoader
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda")
# Training settings
parser = argparse.ArgumentParser(description="PyTorch SRResNet")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--batchSize", type=int, default=2, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=5000, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=1)
parser.add_argument('--gamma', type=float, default=0.98, help='Learning Rate decay')


def l1_loss_with_weight(prediction_results, ground_truth):
    loss_1 = nn.L1Loss()
    loss_2 = nn.MSELoss()
    return loss_1(prediction_results, ground_truth) + 10 * loss_2(prediction_results, ground_truth)


def train():
    global opt, model, netContent
    opt = parser.parse_args()
    print(opt)
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    print("===> Building model")
    model = UNet_2D(inchannels=2, outchannels=1)
    print("===> Setting GPU")
    model = model.cuda()
    model = model.to('cuda')

    num_params = sum(param.numel() for param in model.parameters())
    print(num_params)
    print(model)

    print("===> Setting Optimizer")
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)

    print("===> Training")
    for epoch in range(1, opt.nEpochs + 1):
        print(epoch)
        train_set = TrainSetLoader('/chest', device)
        training_data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize, shuffle=True)
        trainor(training_data_loader, optimizer, model, epoch, scheduler)
        scheduler.step()


def trainor(training_data_loader, optimizer, model, epoch, scheduler):
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    model.train()
    loss_epoch = 0
    for iteration, (raw, img) in enumerate(training_data_loader):
        pre = model(raw)
        # print(pre.shape, raw.shape)
        # view_1 = img.cpu().detach().numpy()[0, 0]
        # view_2 = pre.cpu().detach().numpy()[0, 0]
        #
        # plt.imshow(view_1, cmap="gray")
        # plt.show()
        # plt.imshow(view_2, cmap="gray")
        # plt.show()
        # plt.imshow(view_2 - view_1, cmap="gray")
        # plt.show()
        # plt.imshow(do_filter(view_2 - view_1), cmap="gray")
        # plt.show()

        loss = l1_loss_with_weight(pre, img)
        # ssim = ssim_loss(hr_pre, hr_img)
        # loss = loss + 100 * ssim
        loss_epoch += loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("===> Epoch[{}]: Loss: {:.5f} loss_avg: {:.5f}".format
              (epoch, loss, loss_epoch / (iteration % 100 + 1)))
        if (iteration + 1) % 100 == 0:
            loss_epoch = 0
            save_checkpoint(model, epoch)
            print("model has benn saved")


def save_checkpoint(model, epoch):
    model_out_path = "/data/Model/slice_prediction/" + "chest_epoch_{}.pth".format(epoch)
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


train()


