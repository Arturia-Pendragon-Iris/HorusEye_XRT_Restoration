import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class GetSobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        kernel_v = [[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]]
        kernel_h = [[-1, -2, -1],
                    [0, 0, 0],
                    [1, 2, 1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)
        return x


class GetLaplace(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)

        kernel = [[0, 1, 0],
                  [1, -4, 1],
                  [0, 1, 0]]
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i = F.conv2d(x_i.unsqueeze(1), self.weight, padding=1)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)
        return x


class GetHighPass(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_v = [[-1, -1, -1],
                    [-1, 8, -1],
                    [-1, -1, -1]]
        kernel_h = [[1, -2, 1],
                    [-2, 4, -2],
                    [1, -2, 1]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x


class Getgradientnopadding(nn.Module):
    def __init__(self):
        super(Getgradientnopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False)
        self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False)

    def forward(self, x):
        x_list = []
        for i in range(x.shape[1]):
            x_i = x[:, i]
            x_i_v = F.conv2d(x_i.unsqueeze(1), self.weight_v, padding=1)
            x_i_h = F.conv2d(x_i.unsqueeze(1), self.weight_h, padding=1)
            x_i = torch.sqrt(torch.pow(x_i_v, 2) + torch.pow(x_i_h, 2) + 1e-6)
            x_list.append(x_i)

        x = torch.cat(x_list, dim=1)

        return x


def ln_loss(prediction_results, ground_truth):
    loss_1 = nn.L1Loss()
    loss_2 = nn.MSELoss()
    return (loss_1(prediction_results, ground_truth) +
            torch.sqrt(loss_2(prediction_results, ground_truth))) / 2


def sharp_loss(prediction_results, ground_truth):
    loss_1 = nn.L1Loss()

    get_grad = Getgradientnopadding().cuda()
    get_sobel = GetSobel().cuda()
    get_laplace = GetLaplace().cuda()
    get_high = GetHighPass().cuda()

    loss = 0
    loss += loss_1(get_grad(prediction_results), get_grad(ground_truth))
    loss += loss_1(get_sobel(prediction_results), get_sobel(ground_truth))

    return loss / 2


def grad_loss_simple(prediction_results, ground_truth):
    loss_1 = nn.L1Loss()
    grad_pre = prediction_results[:, :, 1:, 1:, 1:] - prediction_results[:, :, :-1, :-1, :-1]
    grad_gt = ground_truth[:, :, 1:, 1:, 1:] - ground_truth[:, :, :-1, :-1, :-1]
    return loss_1(grad_pre, grad_gt)


def ReconstructionLoss(prediction_results, ground_truth, lamb=0.5, dim="2d"):
    if dim == "2d":
        if lamb != 0:
            return (ln_loss(prediction_results, ground_truth) + lamb * sharp_loss(prediction_results, ground_truth))
        else:
            return ln_loss(prediction_results, ground_truth)
    else:
        # print(prediction_results.permute(0, 4, 2, 3, 1).squeeze(-1).shape,
        #       ground_truth.permute(0, 4, 2, 3, 1).squeeze(-1).shape,)
        if lamb == 0:
            return ln_loss(prediction_results.permute(0, 4, 2, 3, 1).squeeze(-1),
                           ground_truth.permute(0, 4, 2, 3, 1).squeeze(-1))
        else:
            loss_1 = ln_loss(prediction_results.permute(0, 4, 2, 3, 1).squeeze(-1),
                             ground_truth.permute(0, 4, 2, 3, 1).squeeze(-1))
            loss_2 = lamb * sharp_loss(prediction_results.permute(0, 4, 2, 3, 1).squeeze(-1),
                                       ground_truth.permute(0, 4, 2, 3, 1).squeeze(-1))
            # print(loss_1, loss_2)
            return loss_1 + loss_2


def flow_loss(prediction_results, ground_truth, dim="2d"):
    if dim == "2d":
        return ln_loss(prediction_results[:, 1:] - prediction_results[:, :-1],
                       ground_truth[:, 1:] - ground_truth[:, :-1])
    else:
        return ln_loss(prediction_results.permute(0, 4, 2, 3, 1).squeeze(-1)[:, 1:] -
                       prediction_results.permute(0, 4, 2, 3, 1).squeeze(-1)[:, :-1],
                       ground_truth.permute(0, 4, 2, 3, 1).squeeze(-1)[:, 1:] -
                       ground_truth.permute(0, 4, 2, 3, 1).squeeze(-1)[:, :-1])
