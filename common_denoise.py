from skimage.restoration import (
    denoise_tv_chambolle,
    denoise_bilateral,
    denoise_wavelet,
    denoise_nl_means,
    estimate_sigma,
)
from sklearn.feature_extraction import image
from ksvd import ApproximateKSVD
from analysis.filter_2D import gaussian_filter
import cv2
import numpy as np
from sklearn import linear_model
import scipy.misc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import bm3d
import tqdm


def denoise_tv(img, weight=0.1):
    de = denoise_tv_chambolle(img[np.newaxis], weight=weight, channel_axis=0)
    return de[0]


def denoise_bila(img):
    de = denoise_bilateral(img[np.newaxis], sigma_color=0.05, sigma_spatial=15, channel_axis=0)
    return de[0]


def denoise_wave(img):
    de = denoise_wavelet(img[np.newaxis], channel_axis=0, rescale_sigma=True)
    return de[0]


def denoise_wave_scan(img):
    de = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    for i in range(img.shape[-1]):
        de[:, :, i] = denoise_wave(img[:, :, i])
    return np.array(de, "float32")


def denoise_nl(img):
    de = denoise_nl_means(img)
    return de


def denoise_nl_scan(img):
    de = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    for i in range(img.shape[-1]):
        de[:, :, i] = denoise_nl(img[:, :, i])
    return np.array(de, "float32")


def denoise_gaussian(img, sigma=0.5):
    return gaussian_filter(img, sigma=sigma)


def denoise_gaussian_scan(img):
    de = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    for i in range(img.shape[-1]):
        de[:, :, i] = denoise_gaussian(img[:, :, i])
    return np.array(de, "float32")


class KSVD(object):
    def __init__(self, n_components, max_iter=30, tol=5000,
                 n_nonzero_coefs=None):
        """
        稀疏模型Y = DX，Y为样本矩阵，使用KSVD动态更新字典矩阵D和稀疏矩阵X
        :param n_components: 字典所含原子个数（字典的列数）
        :param max_iter: 最大迭代次数
        :param tol: 稀疏表示结果的容差
        :param n_nonzero_coefs: 稀疏度
        """
        self.dictionary = None
        self.sparsecode = None
        self.max_iter = max_iter
        self.tol = tol
        self.n_components = n_components
        self.n_nonzero_coefs = n_nonzero_coefs

    def _initialize(self, y):
        """
        初始化字典矩阵
        """
        u, s, v = np.linalg.svd(y)
        self.dictionary = u[:, :self.n_components]
        # print(self.dictionary.shape)

    def _update_dict(self, y, d, x):
        """
        使用KSVD更新字典的过程
        """
        for i in range(self.n_components):
            index = np.nonzero(x[i, :])[0]
            if len(index) == 0:
                continue

            d[:, i] = 0
            r = (y - np.dot(d, x))[:, index]
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0].T
            x[i, index] = s[0] * v[0, :]
        return d, x

    def fit(self, y):
        """
        KSVD迭代过程
        """
        self._initialize(y)
        for i in range(self.max_iter):
            x = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
            e = np.linalg.norm(y - np.dot(self.dictionary, x))
            if e < self.tol:
                break
            self._update_dict(y, self.dictionary, x)

        self.sparsecode = linear_model.orthogonal_mp(self.dictionary, y, n_nonzero_coefs=self.n_nonzero_coefs)
        return self.dictionary, self.sparsecode


def denoise_ksvd_img(img):
    # ksvd = KSVD(200)
    # dictionary, sparsecode = ksvd.fit(img)
    # output = dictionary.dot(sparsecode)
    # return np.array(output, "float32")

    # patch_size = 8
    # initial_dictionary = DCTDictionary(patch_size, 11)
    # denoiser = KSVDImageDenoiser(initial_dictionary, pursuit=MatchingPursuit)
    # z, d, a = denoiser.denoise(img, sigma=sigma, patch_size=patch_size,
    #                            n_iter=4, multiplier=0.5, noise_gain=1.075)

    patch_size = (5, 5)
    patches = image.extract_patches_2d(img, patch_size)
    signals = patches.reshape(patches.shape[0], -1)
    mean = np.mean(signals, axis=1)[:, np.newaxis]
    signals -= mean
    aksvd = ApproximateKSVD(n_components=32)
    dictionary = aksvd.fit(signals[:1000]).components_
    gamma = aksvd.transform(signals)
    reduced = gamma.dot(dictionary) + mean
    reduced_img = image.reconstruct_from_patches_2d(
        reduced.reshape(patches.shape), img.shape)

    return reduced_img


def denoise_ksvd_scan(img):
    de = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    for i in range(img.shape[-1]):
        de[:, :, i] = denoise_ksvd_img(img[:, :, i])
    return np.array(de, "float32")


cv2.setUseOptimized(True)

# Parameters initialization
sigma = 25
Threshold_Hard3D = 2.7 * sigma  # Threshold for Hard Thresholding
First_Match_threshold = 2500  # 用于计算block之间相似度的阈值
Step1_max_matched_cnt = 16  # 组最大匹配的块数
Step1_Blk_Size = 8  # block_Size即块的大小，8*8
Step1_Blk_Step = 3  # Rather than sliding by one pixel to every next reference block, use a step of Nstep pixels in both horizontal and vertical directions.
Step1_Search_Step = 3  # 块的搜索step
Step1_Search_Window = 39  # Search for candidate matching blocks in a local neighborhood of restricted size NS*NS centered

Second_Match_threshold = 400  # 用于计算block之间相似度的阈值
Step2_max_matched_cnt = 32
Step2_Blk_Size = 8
Step2_Blk_Step = 3
Step2_Search_Step = 3
Step2_Search_Window = 39

Beta_Kaiser = 2.0


def init(img, _blk_size, _Beta_Kaiser):
    """该函数用于初始化，返回用于记录过滤后图像以及权重的数组,还有构造凯撒窗"""
    m_shape = img.shape
    m_img = np.matrix(np.zeros(m_shape, dtype=float))
    m_wight = np.matrix(np.zeros(m_shape, dtype=float))
    K = np.matrix(np.kaiser(_blk_size, _Beta_Kaiser))
    m_Kaiser = np.array(K.T * K)  # 构造一个凯撒窗
    return m_img, m_wight, m_Kaiser


def Locate_blk(i, j, blk_step, block_Size, width, height):
    '''该函数用于保证当前的blk不超出图像范围'''
    if i * blk_step + block_Size < width:
        point_x = i * blk_step
    else:
        point_x = width - block_Size

    if j * blk_step + block_Size < height:
        point_y = j * blk_step
    else:
        point_y = height - block_Size

    m_blockPoint = np.array((point_x, point_y), dtype=int)  # 当前参考图像的顶点

    return m_blockPoint


def Define_SearchWindow(_noisyImg, _BlockPoint, _WindowSize, Blk_Size):
    """该函数返回一个二元组（x,y）,用以界定_Search_Window顶点坐标"""
    point_x = _BlockPoint[0]  # 当前坐标
    point_y = _BlockPoint[1]  # 当前坐标

    # 获得SearchWindow四个顶点的坐标
    LX = point_x + Blk_Size / 2 - _WindowSize / 2  # 左上x
    LY = point_y + Blk_Size / 2 - _WindowSize / 2  # 左上y
    RX = LX + _WindowSize  # 右下x
    RY = LY + _WindowSize  # 右下y

    # 判断一下是否越界
    if LX < 0:
        LX = 0
    elif RX > _noisyImg.shape[0]:
        LX = _noisyImg.shape[0] - _WindowSize
    if LY < 0:
        LY = 0
    elif RY > _noisyImg.shape[0]:
        LY = _noisyImg.shape[0] - _WindowSize

    return np.array((LX, LY), dtype=int)


def Step1_fast_match(_noisyImg, _BlockPoint):
    """快速匹配"""
    '''
    *返回邻域内寻找和当前_block相似度最高的几个block,返回的数组中包含本身
    *_noisyImg:噪声图像
    *_BlockPoint:当前block的坐标及大小
    '''
    (present_x, present_y) = _BlockPoint  # 当前坐标
    Blk_Size = Step1_Blk_Size
    Search_Step = Step1_Search_Step
    Threshold = First_Match_threshold
    max_matched = Step1_max_matched_cnt
    Window_size = Step1_Search_Window

    blk_positions = np.zeros((max_matched, 2), dtype=int)  # 用于记录相似blk的位置
    Final_similar_blocks = np.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)

    img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
    dct_img = cv2.dct(img.astype(np.float64))  # 对目标作block作二维余弦变换

    Final_similar_blocks[0, :, :] = dct_img
    blk_positions[0, :] = _BlockPoint

    Window_location = Define_SearchWindow(_noisyImg, _BlockPoint, Window_size, Blk_Size)
    blk_num = (Window_size - Blk_Size) / Search_Step  # 确定最多可以找到多少相似blk
    blk_num = int(blk_num)
    (present_x, present_y) = Window_location

    similar_blocks = np.zeros((blk_num ** 2, Blk_Size, Blk_Size), dtype=float)
    m_Blkpositions = np.zeros((blk_num ** 2, 2), dtype=int)
    Distances = np.zeros(blk_num ** 2, dtype=float)  # 记录各个blk与它的相似度

    # 开始在_Search_Window中搜索,初始版本先采用遍历搜索策略,这里返回最相似的几块
    matched_cnt = 0
    for i in range(blk_num):
        for j in range(blk_num):
            tem_img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
            dct_Tem_img = cv2.dct(tem_img.astype(np.float64))
            m_Distance = np.linalg.norm((dct_img - dct_Tem_img)) ** 2 / (Blk_Size ** 2)

            # 下面记录数据自动不考虑自身(因为已经记录)
            if m_Distance < Threshold and m_Distance > 0:  # 说明找到了一块符合要求的
                similar_blocks[matched_cnt, :, :] = dct_Tem_img
                m_Blkpositions[matched_cnt, :] = (present_x, present_y)
                Distances[matched_cnt] = m_Distance
                matched_cnt += 1
            present_y += Search_Step
        present_x += Search_Step
        present_y = Window_location[1]
    Distances = Distances[:matched_cnt]
    Sort = Distances.argsort()

    # 统计一下找到了多少相似的blk
    if matched_cnt < max_matched:
        Count = matched_cnt + 1
    else:
        Count = max_matched

    if Count > 0:
        for i in range(1, Count):
            Final_similar_blocks[i, :, :] = similar_blocks[Sort[i - 1], :, :]
            blk_positions[i, :] = m_Blkpositions[Sort[i - 1], :]
    return Final_similar_blocks, blk_positions, Count


def Step1_3DFiltering(_similar_blocks):
    '''
    *3D变换及滤波处理
    *_similar_blocks:相似的一组block,这里已经是频域的表示
    *要将_similar_blocks第三维依次取出,然在频域用阈值滤波之后,再作反变换
    '''
    statis_nonzero = 0  # 非零元素个数
    m_Shape = _similar_blocks.shape

    # 下面这一段代码很耗时
    for i in range(m_Shape[1]):
        for j in range(m_Shape[2]):
            tem_Vct_Trans = cv2.dct(_similar_blocks[:, i, j])
            tem_Vct_Trans[np.abs(tem_Vct_Trans[:]) < Threshold_Hard3D] = 0.
            statis_nonzero += tem_Vct_Trans.nonzero()[0].size
            _similar_blocks[:, i, j] = cv2.idct(tem_Vct_Trans)[0]
    return _similar_blocks, statis_nonzero


def Aggregation_hardthreshold(_similar_blocks, blk_positions, m_basic_img, m_wight_img, _nonzero_num, Count, Kaiser):
    '''
    *对3D变换及滤波后输出的stack进行加权累加,得到初步滤波的图片
    *_similar_blocks:相似的一组block,这里是频域的表示
    *对于最后的数组，乘以凯撒窗之后再输出
    '''
    _shape = _similar_blocks.shape
    if _nonzero_num < 1:
        _nonzero_num = 1
    block_wight = (1. / _nonzero_num) * Kaiser
    for i in range(Count):
        point = blk_positions[i, :]
        tem_img = (1. / _nonzero_num) * cv2.idct(_similar_blocks[i, :, :]) * Kaiser
        m_basic_img[point[0]:point[0] + _shape[1], point[1]:point[1] + _shape[2]] += tem_img
        m_wight_img[point[0]:point[0] + _shape[1], point[1]:point[1] + _shape[2]] += block_wight


def BM3D_1st_step(_noisyImg):
    """第一步,基本去噪"""
    # 初始化一些参数：
    (width, height) = _noisyImg.shape  # 得到图像的长宽
    block_Size = Step1_Blk_Size  # 块大小
    blk_step = Step1_Blk_Step  # N块步长滑动
    Width_num = (width - block_Size) / blk_step
    Height_num = (height - block_Size) / blk_step

    # 初始化几个数组
    Basic_img, m_Wight, m_Kaiser = init(_noisyImg, Step1_Blk_Size, Beta_Kaiser)

    # 开始逐block的处理,+2是为了避免边缘上不够
    for i in range(int(Width_num + 2)):
        for j in range(int(Height_num + 2)):
            # m_blockPoint当前参考图像的顶点
            m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height)  # 该函数用于保证当前的blk不超出图像范围
            Similar_Blks, Positions, Count = Step1_fast_match(_noisyImg, m_blockPoint)
            Similar_Blks, statis_nonzero = Step1_3DFiltering(Similar_Blks)
            Aggregation_hardthreshold(Similar_Blks, Positions, Basic_img, m_Wight, statis_nonzero, Count, m_Kaiser)
    Basic_img[:, :] /= m_Wight[:, :]
    basic = np.matrix(Basic_img, dtype=int)
    basic.astype(np.uint8)

    return basic


def Step2_fast_match(_Basic_img, _noisyImg, _BlockPoint):
    '''
    *快速匹配算法,返回邻域内寻找和当前_block相似度最高的几个block,要同时返回basicImg和IMG
    *_Basic_img: 基础去噪之后的图像
    *_noisyImg:噪声图像
    *_BlockPoint:当前block的坐标及大小
    '''
    (present_x, present_y) = _BlockPoint  # 当前坐标
    Blk_Size = Step2_Blk_Size
    Threshold = Second_Match_threshold
    Search_Step = Step2_Search_Step
    max_matched = Step2_max_matched_cnt
    Window_size = Step2_Search_Window

    blk_positions = np.zeros((max_matched, 2), dtype=int)  # 用于记录相似blk的位置
    Final_similar_blocks = np.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)
    Final_noisy_blocks = np.zeros((max_matched, Blk_Size, Blk_Size), dtype=float)

    img = _Basic_img[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
    dct_img = cv2.dct(img.astype(np.float32))  # 对目标作block作二维余弦变换
    Final_similar_blocks[0, :, :] = dct_img

    n_img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
    dct_n_img = cv2.dct(n_img.astype(np.float32))  # 对目标作block作二维余弦变换
    Final_noisy_blocks[0, :, :] = dct_n_img

    blk_positions[0, :] = _BlockPoint

    Window_location = Define_SearchWindow(_noisyImg, _BlockPoint, Window_size, Blk_Size)
    blk_num = (Window_size - Blk_Size) / Search_Step  # 确定最多可以找到多少相似blk
    blk_num = int(blk_num)
    (present_x, present_y) = Window_location

    similar_blocks = np.zeros((blk_num ** 2, Blk_Size, Blk_Size), dtype=float)
    m_Blkpositions = np.zeros((blk_num ** 2, 2), dtype=int)
    Distances = np.zeros(blk_num ** 2, dtype=float)  # 记录各个blk与它的相似度

    # 开始在_Search_Window中搜索,初始版本先采用遍历搜索策略,这里返回最相似的几块
    matched_cnt = 0
    for i in range(blk_num):
        for j in range(blk_num):
            tem_img = _Basic_img[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
            dct_Tem_img = cv2.dct(tem_img.astype(np.float32))
            m_Distance = np.linalg.norm((dct_img - dct_Tem_img)) ** 2 / (Blk_Size ** 2)

            # 下面记录数据自动不考虑自身(因为已经记录)
            if m_Distance < Threshold and m_Distance > 0:
                similar_blocks[matched_cnt, :, :] = dct_Tem_img
                m_Blkpositions[matched_cnt, :] = (present_x, present_y)
                Distances[matched_cnt] = m_Distance
                matched_cnt += 1
            present_y += Search_Step
        present_x += Search_Step
        present_y = Window_location[1]
    Distances = Distances[:matched_cnt]
    Sort = Distances.argsort()

    # 统计一下找到了多少相似的blk
    if matched_cnt < max_matched:
        Count = matched_cnt + 1
    else:
        Count = max_matched

    if Count > 0:
        for i in range(1, Count):
            Final_similar_blocks[i, :, :] = similar_blocks[Sort[i - 1], :, :]
            blk_positions[i, :] = m_Blkpositions[Sort[i - 1], :]

            (present_x, present_y) = m_Blkpositions[Sort[i - 1], :]
            n_img = _noisyImg[present_x: present_x + Blk_Size, present_y: present_y + Blk_Size]
            Final_noisy_blocks[i, :, :] = cv2.dct(n_img.astype(np.float64))

    return Final_similar_blocks, Final_noisy_blocks, blk_positions, Count


def Step2_3DFiltering(_Similar_Bscs, _Similar_Imgs):
    '''
    *3D维纳变换的协同滤波
    *_similar_blocks:相似的一组block,这里是频域的表示
    *要将_similar_blocks第三维依次取出,然后作dct,在频域进行维纳滤波之后,再作反变换
    *返回的Wiener_wight用于后面Aggregation
    '''
    m_Shape = _Similar_Bscs.shape
    Wiener_wight = np.zeros((m_Shape[1], m_Shape[2]), dtype=float)

    for i in range(m_Shape[1]):
        for j in range(m_Shape[2]):
            tem_vector = _Similar_Bscs[:, i, j]
            tem_Vct_Trans = np.matrix(cv2.dct(tem_vector))
            Norm_2 = np.float64(tem_Vct_Trans.T * tem_Vct_Trans)
            m_weight = Norm_2 / (Norm_2 + sigma ** 2)
            if m_weight != 0:
                Wiener_wight[i, j] = 1. / (m_weight ** 2 * sigma ** 2)
            # else:
            #     Wiener_wight[i, j] = 10000
            tem_vector = _Similar_Imgs[:, i, j]
            tem_Vct_Trans = m_weight * cv2.dct(tem_vector)
            _Similar_Bscs[:, i, j] = cv2.idct(tem_Vct_Trans)[0]

    return _Similar_Bscs, Wiener_wight


def Aggregation_Wiener(_Similar_Blks, _Wiener_wight, blk_positions, m_basic_img, m_wight_img, Count, Kaiser):
    '''
    *对3D变换及滤波后输出的stack进行加权累加,得到初步滤波的图片
    *_similar_blocks:相似的一组block,这里是频域的表示
    *对于最后的数组，乘以凯撒窗之后再输出
    '''
    _shape = _Similar_Blks.shape
    block_wight = _Wiener_wight  # * Kaiser

    for i in range(Count):
        point = blk_positions[i, :]
        tem_img = _Wiener_wight * cv2.idct(_Similar_Blks[i, :, :])  # * Kaiser
        m_basic_img[point[0]:point[0] + _shape[1], point[1]:point[1] + _shape[2]] += tem_img
        m_wight_img[point[0]:point[0] + _shape[1], point[1]:point[1] + _shape[2]] += block_wight


def BM3D_2nd_step(_basicImg, _noisyImg):
    '''Step 2. 最终的估计: 利用基本的估计，进行改进了的分组以及协同维纳滤波'''
    # 初始化一些参数：
    (width, height) = _noisyImg.shape
    block_Size = Step2_Blk_Size
    blk_step = Step2_Blk_Step
    Width_num = (width - block_Size) / blk_step
    Height_num = (height - block_Size) / blk_step

    # 初始化几个数组
    m_img, m_Wight, m_Kaiser = init(_noisyImg, block_Size, Beta_Kaiser)

    for i in range(int(Width_num + 2)):
        for j in range(int(Height_num + 2)):
            m_blockPoint = Locate_blk(i, j, blk_step, block_Size, width, height)
            Similar_Blks, Similar_Imgs, Positions, Count = Step2_fast_match(_basicImg, _noisyImg, m_blockPoint)
            Similar_Blks, Wiener_wight = Step2_3DFiltering(Similar_Blks, Similar_Imgs)
            Aggregation_Wiener(Similar_Blks, Wiener_wight, Positions, m_img, m_Wight, Count, m_Kaiser)
    m_img[:, :] /= m_Wight[:, :]
    Final = np.matrix(m_img, dtype=int)
    Final.astype(np.uint8)

    return Final


def denoise_bm3d(img, var=1):
    return bm3d.bm3d(img, sigma_psd=var)


def denoise_bm3d_scan(img, var):
    de = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    for i in range(img.shape[-1]):
        de[:, :, i] = denoise_bm3d(img[:, :, i], var)
    return np.array(de, "float32")


class network(nn.Module):
    def __init__(self, n_chan, chan_embed=48):
        super(network, self).__init__()

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv1 = nn.Conv2d(n_chan, chan_embed, 3, padding=1)
        self.conv2 = nn.Conv2d(chan_embed, chan_embed, 3, padding=1)
        self.conv3 = nn.Conv2d(chan_embed, n_chan, 1)

    def forward(self, x):
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.conv3(x)

        return x


def pair_downsampler(img):
    # img has shape B C H W
    c = img.shape[1]

    filter1 = torch.FloatTensor([[[[0, 0.5], [0.5, 0]]]]).to(img.device)
    filter1 = filter1.repeat(c, 1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5, 0], [0, 0.5]]]]).to(img.device)
    filter2 = filter2.repeat(c, 1, 1, 1)

    output1 = F.conv2d(img, filter1, stride=2, groups=c)
    output2 = F.conv2d(img, filter2, stride=2, groups=c)

    return output1, output2


def mse(gt: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt, pred)


def loss_func(noisy_img, model):
    noisy1, noisy2 = pair_downsampler(noisy_img)

    pred1 = noisy1 - model(noisy1)
    pred2 = noisy2 - model(noisy2)

    loss_res = 1 / 2 * (mse(noisy1, pred2) + mse(noisy2, pred1))

    noisy_denoised = noisy_img - model(noisy_img)
    denoised1, denoised2 = pair_downsampler(noisy_denoised)

    loss_cons = 1 / 2 * (mse(pred1, denoised1) + mse(pred2, denoised2))

    loss = loss_res + loss_cons

    return loss


def train(model, optimizer, noisy_img):
    loss = loss_func(noisy_img, model)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def denoise(model, noisy_img):
    with torch.no_grad():
        pred = torch.clamp(noisy_img - model(noisy_img), 0, 1)

    return pred


def denoise_2noise_img(img):
    img = np.clip(img, 0, 1)
    img = torch.tensor(img[np.newaxis, np.newaxis]).cuda().to(torch.float)
    model = network(n_chan=1)
    model = model.cuda()

    max_epoch = 2000  # training epochs
    lr = 0.002  # learning rate
    step_size = 50  # number of epochs at which learning rate decays
    gamma = 0.5  # factor by which learning rate decays

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    for epoch in range(max_epoch):
        train(model, optimizer, img)
        scheduler.step()

    denoised_img = denoise(model, img)

    denoised = denoised_img.cpu().detach().numpy()[0, 0]
    return denoised


def denoise_2noise_scan(img):
    de = np.zeros([img.shape[0], img.shape[1], img.shape[2]])
    for i in range(img.shape[-1]):
        de[:, :, i] = denoise_2noise_img(img[:, :, i])
    return np.array(de, "float32")
