import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from skimage.morphology import skeletonize_3d
from skimage.measure import label
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import math


def branch_detected_calculation(pred, label_parsing, label_skeleton, thresh=0.8):
    label_branch = label_skeleton * label_parsing
    label_branch_flat = label_branch.flatten()
    label_branch_bincount = np.bincount(label_branch_flat)[1:]
    total_branch_num = label_branch_bincount.shape[0]
    pred_branch = label_branch * pred
    pred_branch_flat = pred_branch.flatten()
    pred_branch_bincount = np.bincount(pred_branch_flat)[1:]
    if total_branch_num != pred_branch_bincount.shape[0]:
        lack_num = total_branch_num - pred_branch_bincount.shape[0]
        pred_branch_bincount = np.concatenate((pred_branch_bincount, np.zeros(lack_num)))
    branch_ratio_array = pred_branch_bincount / label_branch_bincount
    branch_ratio_array = np.where(branch_ratio_array >= thresh, 1, 0)
    detected_branch_num = np.count_nonzero(branch_ratio_array)
    detected_branch_ratio = round((detected_branch_num * 100) / total_branch_num, 2)
    return total_branch_num, detected_branch_num, detected_branch_ratio


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


def count_branches(blood_vessel, lung):
    center = np.array(skeletonize_3d(blood_vessel), "float32")
    center = np.array(center > 0.1, "float32")
    # center = np.array(skeletonize_3d(center), "float32")
    # center = np.array(center > 0.1, "float32")

    # center += get_surface_3D(center * lung, outer=True)
    # visualize_numpy_as_stl(center)
    center = center * lung
    length = np.sum(center)
    # center = np.sum(center * lung)
    loc = np.array(np.where(center > 0))
    node = np.zeros(blood_vessel.shape)
    for i in range(loc.shape[-1]):
        [x, y, z] = loc[:, i]
        # print(np.sum(center[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]))
        # if np.sum(center[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]) <= 2.1:
        #     jishu += 1
        if np.sum(center[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]) >= 3:
            node[x, y, z] = 1

    # node += get_surface_3D(node, outer=True, strict=True)
    # visualize_one_numpy(node)
    _, branch = label(node, connectivity=2, return_num=True)

    # node = np.array()

    return length, branch


def count_branches_1(blood_vessel):
    center = np.array(skeletonize_3d(blood_vessel), "float32")
    center = np.array(center > 0.1, "float32")

    loc = np.array(np.where(center > 0))
    node = np.zeros(blood_vessel.shape)
    for i in range(loc.shape[-1]):
        [x, y, z] = loc[:, i]

        if np.sum(center[x - 1:x + 2, y - 1:y + 2, z - 1:z + 2]) == 2:
            node[x, y, z] = 1

    # node += get_surface_3D(node, outer=True, strict=True)
    # visualize_one_numpy(node)
    # _, branch = label(node, connectivity=2, return_num=True)
    branch = np.sum(node)
    # node = np.array()

    return branch


def compute_auc(pre, label, lung=None):
    # if not lung is None:
    auc = roc_auc_score(label.reshape(-1), pre.reshape(-1))
    auc = round(auc, 3)
    return auc


def compute_snr_v0(noise_img, clean_img):
    std_1 = np.var(clean_img)
    std_2 = np.var(clean_img - noise_img)

    snr = 10 * (np.log10(std_1) - np.log10(std_2))
    return snr


def compute_snr_v1(noise_img, clean_img, background):
    snr = 0
    for i in range(len(background)):
        region_clean = clean_img[background[i][0]:background[i][2], background[i][1]:background[i][3]]
        region_noise = noise_img[background[i][0]:background[i][2], background[i][1]:background[i][3]]

        v_b = np.var(region_noise)
        s = (np.mean(region_clean) - np.mean(region_noise)) ** 2

        snr += 10 / len(background) * (np.log10(s) - np.log10(v_b))

    return snr


def compute_snr_v2(img, background):
    region = img[background[0]:background[2], background[1]:background[3]]

    v_b = np.std(region)
    s = np.mean(region)
    return (1 - s) / v_b
