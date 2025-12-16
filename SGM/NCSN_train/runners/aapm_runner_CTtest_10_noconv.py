import os
import numpy as np
import torch
import torch.nn as nn
import math
print(1)
from NCSN_train.models.cond_refinenet_dilated_noconv import CondRefineNetDilated
print(1)
from torchvision.datasets import MNIST, CIFAR10
from torchvision import transforms
from torch.utils.data import DataLoader
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr,structural_similarity as compare_ssim,mean_squared_error as compare_mse
from skimage.restoration import denoise_tv_chambolle, denoise_wavelet
import glob
import time
from scipy.io import loadmat,savemat
#from scipy.misc import imread,imsave
import imageio
from scipy.linalg import norm,orth
from scipy.stats import poisson
#import dicom
import pydicom as dicom
from skimage.transform import radon, iradon
print(1)
import odl
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
print(1)
import scipy.misc
import matplotlib


plt.ion()
# Please, run here to test
# python separate_ImageNet.py --model ncsn --runner Aapm_Runner_CTtest_10_noconv --config aapm_10C.yml --doc AapmCT_10C --test --image_folder output
__all__ = ['Aapm_Runner_CTtest_10_noconv']

class GetCT(Dataset):

    def __init__(self,root,augment=None):
        super().__init__()
        self.data_names = np.array([root+"/"+x  for x in os.listdir(root)])

        self.augment = None

    def __getitem__(self,index):


        dataCT=dicom.read_file(self.data_names[index])
        
        
        return dataCT
    
    def __len__(self):

            return len(self.data_names)
class Aapm_Runner_CTtest_10_noconv():
    def __init__(self, args, config):
        self.args = args
        self.config = config
    def write_images(self,x,image_save_path):
        x = np.array(x,dtype=np.uint8)
        cv2.imwrite(image_save_path, x)

    def test(self):
        N = 512
        ANG = 180
        VIEW = 360
        cols = rows =512
        THETA = np.linspace(0, ANG, VIEW + 1)
        THETA = THETA[:-1]
        # angle
        angle_partition = odl.uniform_partition(0, 2 * np.pi, 1000)
        detector_partition = odl.uniform_partition(-360, 360, 1000)
        geometry = odl.tomo.FanBeamGeometry(angle_partition, detector_partition,
                                    src_radius=500, det_radius=500)
        reco_space = odl.uniform_discr(min_pt=[-128, -128], max_pt=[128, 128], shape=[512, 512], dtype='float32')
        ray_trafo = odl.tomo.RayTransform(reco_space, geometry)
        pseudoinverse = odl.tomo.fbp_op(ray_trafo)

        def TV_J(real):
            import numpy as np
            from matplotlib import pyplot as plt
            from matplotlib import gridspec
            import matplotlib.pyplot as plt
            from skimage.data import chelsea, hubble_deep_field
            from skimage.metrics import mean_squared_error as mse
            from skimage.metrics import peak_signal_noise_ratio as psnr
            from skimage.restoration import (calibrate_denoiser,
                                            denoise_wavelet,
                                            denoise_tv_chambolle, denoise_nl_means,
                                            estimate_sigma)
            from skimage.util import img_as_float, random_noise
            from skimage.color import rgb2gray
            from functools import partial
            from skimage.metrics import peak_signal_noise_ratio, structural_similarity
            _denoise_wavelet = partial(denoise_wavelet, rescale_sigma=True)
            # Parameters to test when calibrating the denoising algorithm
            parameter_ranges = {'sigma': np.linspace(0.001, 0.0000001,15),
                            'wavelet': ['haar','sym2'],
                            'mode':['soft'],
                            'wavelet_levels':[2],
                            'method':['BayesShrink','VisuShrink']}

            # Denoised image using default parameters of `denoise_wavelet`
            default_output = denoise_wavelet(real)

            # Calibrate denoiser
            calibrated_denoiser = calibrate_denoiser(real,
                                                    _denoise_wavelet,
                                                    denoise_parameters=parameter_ranges
                                                    )

            # Denoised image using calibrated denoiser
            calibrated_output = calibrated_denoiser(real)
            return calibrated_output
        
   


        save_path='./NCSN_train/L506_rec_5e3/'
  
        ## data load

        dataset = dicom.read_file('./testcase1.IMA')
    
        
        img1 = dataset.pixel_array.astype(np.float32)
        img = img1
        RescaleSlope = dataset.RescaleSlope
        RescaleIntercept = dataset.RescaleIntercept     
        CT_img = img * RescaleSlope + RescaleIntercept
        
        # Load the score network
    
        states = torch.load(os.path.join('./checkpoint_5e3_github.pth'), map_location=self.config.device)
        print(os.path.join(self.args.log))
        scorenet = CondRefineNetDilated(self.config).to(self.config.device)
        scorenet = torch.nn.DataParallel(scorenet, device_ids=[0])
        ###############################
        model_dict = scorenet.state_dict()
        # Delete keys in pretrained_dict that are not model_dict
        pretrained_dict = {k: v for k, v in states[0].items() if k in model_dict}
        model_dict.update(pretrained_dict)
        
        scorenet.load_state_dict(model_dict)
        print("split module.wcnn")
        ###############################

        scorenet.eval()

        ## degrade process
        pre_img = (CT_img+1000)/1000*0.02
        ATA = ray_trafo.adjoint(ray_trafo(ray_trafo.domain.one()))
        # print(ATA)
        ## LOW-DOSE SINOGRAM GENERATION
        photons_per_pixel =  5e3
        mu_water = 0.02
        phantom = reco_space.element(img)
        phantom = phantom/1000.0
      
        proj_data = ray_trafo(phantom)
        proj_data = np.exp(-proj_data * mu_water)
        proj_data = odl.phantom.poisson_noise(proj_data * photons_per_pixel)
        proj_data = np.maximum(proj_data, 1) / photons_per_pixel
        proj_data = np.log(proj_data) * (-1 / mu_water)
        image_input = pseudoinverse(proj_data)
        image_input = image_input
        saveinput = np.array(image_input)
        x = np.copy(image_input)
        z = np.copy(x)
        maxdegrade = np.max(image_input)
        image_input = image_input.asarray()
    
        image_gt = (CT_img-np.min(CT_img))/(np.max(CT_img)-np.min(CT_img))
        image_shape = list((1,)+(10,)+image_input.shape[0:2])
        x0 = nn.Parameter(torch.Tensor(np.zeros(image_shape)).uniform_(-1,1)).cuda()

        x01 = x0
        step_lr=0.6*0.00003
        sigmas = np.exp(np.linspace(np.log(1), np.log(0.01),12))
        n_steps_each = 100
        max_psnr = 0
        max_ssim = 0
        min_psnr = 50
        min_ssim = 0.99
        n = 0
        m = 0
        start_start = time.time()
        with torch.no_grad():
            for idx, sigma in enumerate(sigmas):
                start_out = time.time()
                print(idx)
                lambda_recon = 1./sigma**2
                labels = torch.ones(1, device=x0.device) * idx
                labels = labels.long()
                step_size = step_lr * (sigma / sigmas[-1]) ** 2
                print('sigma = {}'.format(sigma))
                ### SQS
                for step in range(n_steps_each):
                    start_in = time.time()
                    noise1 = torch.rand_like(x0)* np.sqrt(step_size * 2)
                    grad1 = scorenet(x01, labels).detach()
                    x0 = x0 + step_size * grad1
                    x01 = x0 + noise1
                    #print(step_size)
                    x0=np.array(x0.cpu().detach(),dtype = np.float32)
                    x1 = np.squeeze(x0)
                    x1 = np.mean(x1,axis=0)
                    
                    im = x1
                    psnr1 = compare_psnr(255*abs(x1),255*abs(image_gt),data_range=255)
                    ssim1 = compare_ssim(abs(x1),abs(image_gt),data_range=1)
                    ## ********** SQS ********* ##
                    hyper = 350
                    sum_diff = x - x1*maxdegrade
                    
                    norm_diff = ray_trafo.adjoint((ray_trafo(x) - proj_data))
                    x_new = z - (norm_diff + 2*hyper*sum_diff)/(ATA + 2*hyper)
                    z = x_new + 0.5 * (x_new - x)
                    x = x_new
                    x_rec = x.asarray()
                    x_rec = x_rec/maxdegrade
                    # The regularization constraint of low-dose CT is small, 
                    # and the sparse Angle task requires larger regularization constraint
                    # Set TV iteration
                    if (step%5)==0:
                        x_rec1 = TV_J(x_rec)
                        x_rec = x_rec+sigma*(x_rec1-x_rec)
               
                        
                  
                    # Select by PSNR or SSIM indicator

                    psnr2 = compare_psnr(255*abs(x_rec),255*abs(image_gt),data_range=255)
                    ssim2 = compare_ssim(abs(x_rec),abs(image_gt),data_range=1)
                    end_in = time.time()
                    #print("inner loop:%.2fs"%(end_in-start_in))
                    #print(x_rec.shape,image_gt.shape)
                    psnr = compare_psnr(255*x_rec,255*(image_gt*np.max(phantom)/maxdegrade),data_range=256)
                    ssim = compare_ssim(255*x_rec,255*(image_gt*np.max(phantom)/maxdegrade),data_range=256)
                    if ssim > max_ssim: 
                        max_ssim = ssim
                        out = x_rec
                    if ssim <min_ssim:
                        min_ssim = ssim
                    print("current {} step".format(step),'PSNR :', psnr,'SSIM :', ssim)
                    x_mid = np.zeros([1,10,512,512],dtype=np.float32)
                    x_rec = np.clip(x_rec,0,1)
                    x_rec = np.expand_dims(x_rec,2)
                    x_mid_1 = np.tile(x_rec,[1,1,10])
                    x_mid_1 = np.transpose(x_mid_1,[2,0,1])
                    x_mid[0,:,:,:] = x_mid_1
                    x0 = torch.tensor(x_mid,dtype=torch.float32).cuda()
                end_out = time.time()
             
                print("outer iter:%.2fs"%(end_out-start_out))
                end_end = time.time()
            savemat(save_path+'wcnnLDCT.mat',{'input':saveinput,'recon': out,'label': image_gt*np.max(phantom)/maxdegrade})
            
            del x, x0, x01, x1, z, x_rec, out, image_input, proj_data
       
            
            

        