import torch
import torch.nn as nn
import numpy as np
import pydicom
import torch.nn.functional as F

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

class RED(nn.Module):
    def __init__(self, out_ch=64):
        super(RED, self).__init__()
        self.conv1 = nn.Conv2d(1 + 1, out_ch, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_ch, out_ch * 2, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(out_ch * 2, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.conv4 = nn.Conv2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.conv5 = nn.Conv2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)

        self.tconv1 = nn.ConvTranspose2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.tconv2 = nn.ConvTranspose2d(out_ch * 4, out_ch * 4, kernel_size=5, stride=1, padding=0)
        self.tconv3 = nn.ConvTranspose2d(out_ch * 4, out_ch * 2, kernel_size=5, stride=1, padding=0)
        self.tconv4 = nn.ConvTranspose2d(out_ch * 2, out_ch, kernel_size=5, stride=1, padding=0)
        self.tconv5 = nn.ConvTranspose2d(out_ch, 1, kernel_size=5, stride=1, padding=0)
        self.get_g_nopadding = Getgradientnopadding()
        self.relu = nn.ReLU()

    def forward(self, x):
        # encoder
        grad = self.get_g_nopadding(x)

        residual_1 = x
        out = self.relu(self.conv1(torch.concatenate((x, grad), dim=1)))
        out = self.relu(self.conv2(out))
        residual_2 = out
        out = self.relu(self.conv3(out))
        out = self.relu(self.conv4(out))
        residual_3 = out
        out = self.relu(self.conv5(out))
        # decoder
        out = self.tconv1(out)
        out += residual_3
        out = self.tconv2(self.relu(out))
        out = self.tconv3(self.relu(out))
        out += residual_2
        out = self.tconv4(self.relu(out))
        out = self.tconv5(self.relu(out))
        out = residual_1 + out
        return out


def predict_denoised_slice(ct_slice, model_path):
    model = RED(out_ch=32)
    model = model.cuda()
    model.half()
    model.eval()
    slice = torch.tensor(ct_slice[np.newaxis, np.newaxis, :]).to(torch.float).to('cuda').half()
    model.load_state_dict(torch.load(model_path))
    # model.load_state_dict(torch.load(
    #     "/data/Model/denoise_V9/RED_200_grad.pth"))
    denoised = model(slice).cpu().detach().numpy()[0, 0]
    return np.array(denoised, "float32")


def predict_dicom(dicom_file, window="lung", model_path='./model/RED_1600_v2.pth'):
    dcm = pydicom.read_file(dicom_file)
    ct_data = np.array(dcm.pixel_array, "float32") - 1000
    assert window in ["lung", "mediastinal", "abdomen"]
    if window == "lung":
        ct_data = np.clip((ct_data + 1000) / 1400, 0, 1)
    elif window == "mediastinal":
        ct_data = np.clip((ct_data + 200) / 500, 0, 1)
    else:
        ct_data = np.clip((ct_data + 160) / 400, 0, 1)

    de = np.clip(predict_denoised_slice(ct_data, model_path), 0, 1)

    if window == "lung":
        de = np.array(de * 1400 - 1000, "int32")
    elif window == "mediastinal":
        de = np.array(de * 500 - 200, "int32")
    else:
        de = np.array(de * 400 - 160, "int32")
    print(ct_data, de)
    return de

def api_gui(dcm, window="lung", model_path='./model/RED_1600_v2.pth'):
    ct_data = np.array(dcm.pixel_array, "float32") - 1000
    assert window in ["lung", "mediastinal", "abdomen"]
    if window == "lung":
        ct_data = np.clip((ct_data + 1000) / 1400, 0, 1)
    elif window == "mediastinal":
        ct_data = np.clip((ct_data + 200) / 500, 0, 1)
    else:
        ct_data = np.clip((ct_data + 160) / 400, 0, 1)

    de = np.clip(predict_denoised_slice(ct_data, model_path), 0, 1)

    if window == "lung":
        de = np.array(de * 1400 - 1000, "int32")
    elif window == "mediastinal":
        de = np.array(de * 500 - 200, "int32")
    else:
        de = np.array(de * 400 - 160, "int32")
    return de

if __name__ == '__main__':
    predict_dicom('匿名.Seq4.Ser203.Img94.dcm')