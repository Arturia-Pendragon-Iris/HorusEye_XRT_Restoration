import argparse, os
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import numpy as np
import torch.optim as optim
from torch.utils.data import DataLoader
from temp.models import RED
from utils import TrainSetLoader
from losses import reconstruction_loss

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0")
# Training settings
parser = argparse.ArgumentParser(description="HorusEye")
parser.add_argument("--cuda", action="store_false", help="Use cuda?")
parser.add_argument("--batchSize", type=int, default=4, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=3, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=2e-5, help="Learning Rate. Default=1e-4")
parser.add_argument("--step", type=int, default=1)
parser.add_argument("--threads", type=int, default=8, help="Number of threads for data loader to use, Default: 1")
parser.add_argument('--gamma', type=float, default=0.6, help='Learning Rate decay')


def train():
    opt = parser.parse_args()
    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    cudnn.benchmark = True

    denoise_model = RED(out_ch=64)
    pretrained_model = torch.load("/data/Model/denoise_V10/RED_v2_0.15.pth")
    denoise_model.load_state_dict(pretrained_model, strict=True)
    denoise_model = denoise_model.cuda()

    num_params = sum(param.numel() for param in denoise_model.parameters())
    print(num_params)
    print(denoise_model)

    print("===> Setting Optimizer")
    denoise_optimizer = optim.AdamW(denoise_model.parameters(), lr=opt.lr)
    denoise_scheduler = torch.optim.lr_scheduler.StepLR(denoise_optimizer, step_size=opt.step, gamma=opt.gamma)

    print("===> Training")
    for epoch in range(1, opt.nEpochs + 1):
        print(epoch)
        training_set = TrainSetLoader('/clean_CT', device)
        training_loader = DataLoader(dataset=training_set, num_workers=0, batch_size=opt.batchSize, shuffle=True)
        trainor(training_loader, denoise_optimizer, denoise_model, epoch)
        denoise_scheduler.step()
        # seg_scheduler.step()


def trainor(training_loader, denoise_optimizer, denoise_model, epoch):
    print("Epoch={}, lr={}".format(epoch, denoise_optimizer.param_groups[0]["lr"]))

    denoise_model.train()
    loss_epoch = 0
    for iteration, (clean, noisy) in enumerate(training_loader):

        # clean = m_model(raw)
        # noisy = noise + clean

        denoise_pre = denoise_model(noisy)
        loss = reconstruction_loss(denoise_pre, clean, lamb=0.1)
        denoise_optimizer.zero_grad()
        loss.backward()
        denoise_optimizer.step()
        loss_epoch += loss

        print("===> Epoch[{}]: loss: {:.5f}  avg_loss: {:.5f}".format
              (epoch, loss, loss_epoch / (iteration % 200 + 1)))

        if (iteration + 1) % 100 == 0:
            loss_epoch = 0
            save_checkpoint(denoise_model, epoch, "/data/Model/denoise_V10/")
            print("model has benn saved")


def save_checkpoint(model, epoch, path):
    model_out_path = os.path.join(path, "RED_v2_0.10.pth")
    torch.save(model.state_dict(), model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


if __name__ == '__main__':
    # torch.multiprocessing.set_start_method("spawn")
    train()


