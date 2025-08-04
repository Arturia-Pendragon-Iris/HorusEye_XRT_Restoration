import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage.morphology import skeletonize_3d
from skimage.measure import label
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import math

def dice_score(pred, label, smooth=1e-5):
    pred = np.array(pred > 0.5, "float32")
    label = np.array(label > 0.5, "float32")
    # pred = pred.flatten()
    # label = label.flatten()
    intersection = np.sum(pred * label)
    dice_coefficient_score = round(((2.0 * intersection + smooth) / (np.sum(pred) + np.sum(label) + smooth)) * 100, 2)
    return dice_coefficient_score


def tree_length_calculation(pred, label_skeleton, smooth=1e-5):
    # pred = pred.flatten()
    # label_skeleton = label_skeleton.flatten()
    tree_length = round((np.sum(pred * label_skeleton) + smooth) / (np.sum(label_skeleton) + smooth) * 100, 2)
    return tree_length


def false_positive(pred, label, smooth=1e-5):
    # pred = pred.flatten()
    # label = label.flatten()
    fp = np.sum(pred - pred * label) + smooth
    fpr = round(fp * 100 / (np.sum((1.0 - label)) + smooth), 3)
    return fpr


def false_negative(pred, label, smooth=1e-5):
    # pred = pred.flatten()
    # label = label.flatten()
    fn = np.sum(label - pred * label) + smooth
    fnr = round(fn * 100 / (np.sum(label) + smooth), 3)
    return fnr


def true_negative(pred, label):
    # specificity or True negative
    sensitivity = round(100 - false_positive(pred, label), 3)
    return sensitivity


def true_positive(pred, label):
    # sensitivity or True Positive
    specificity = round(100 - false_negative(pred, label), 3)
    return specificity


def mis_classification(pred_1, pred_2, label_1, label_2):
    # specificity or True negative
    mis_1 = np.sum(pred_1 * label_2) / np.sum(label_2 * label_2)
    mis_2 = np.sum(pred_2 * label_1) / np.sum(label_1 * label_1)
    mis = round((mis_1 + mis_2) / 2 * 100, 3)
    return mis


def get_roc_value(pre, label, show=True):
    roc_list = []
    for thre in np.arange(0, 0.05, 0.01):
        sub_pre = np.array(pre > thre, "float32")
        fpr = false_positive(sub_pre, label)
        tpr = true_positive(sub_pre, label)
        roc_list.append([round(thre, 2), tpr, fpr])

        # print(len(roc_list))
    if show:
        for i in range(len(roc_list)):
            print(roc_list[i][0], roc_list[i][1], roc_list[i][2])
    return roc_list


def precision(pred, label, smooth=1e-5):
    pred = pred.flatten()
    label = label.flatten()
    tp = np.sum(pred * label) + smooth
    precisions = round(tp * 100 / (np.sum(pred) + smooth), 3)
    return precisions


def normalize(img, n_max=600, n_min=-1000):
    return np.clip((img - n_min) / (n_max - n_min), 0, 1)


def compare_img(img_1, img_2, norm=False):
    if not norm:
        img_1 = np.array(np.clip(img_1, 0, 1), "float32")
        img_2 = np.array(np.clip(img_2, 0, 1), "float32")
    else:
        # img_1 = normalize(img_1)
        # img_2 = normalize(img_2)
        img_1 = (img_1 - np.min(img_1)) / (np.max(img_1) - np.min(img_1))
        img_2 = (img_2 - np.min(img_2)) / (np.max(img_2) - np.min(img_2))

    psnr = round(peak_signal_noise_ratio(img_1, img_2), 4)
    ssim = round(structural_similarity(img_1, img_2, data_range=1), 4)

    nmse = round(np.sqrt(np.mean(np.square(img_1 - img_2)) / np.mean(np.square(img_1))), 4)
    # mse = np.sqrt(mse)

    nmae = round(np.mean(np.abs(img_1 - img_2)) / np.mean(np.abs(img_1)), 4)
    return psnr, ssim, nmse, nmae


def compute_auc(pre, label, lung=None):
    # if not lung is None:
    auc = roc_auc_score(label.reshape(-1), pre.reshape(-1))
    auc = round(auc, 3)
    return auc
