# HorusEye for generalizable X-ray tomography restoration
This is the official repository of "HorusEye: A self-supervised foundation model for generalizable X-ray tomography restoration" by Yuetan Chu , Longxi Zhou , Gongning Luo , Kai Kang , Suyu Dong , Zhongyi Han , Lianming Wu , Xianglin Meng , Changchun Yang , Xin Guo , Yuan Cheng , Yuan Qi , Xin Liu , Dexuan Xie , Ricardo Henao , Anthony Capone , Xigang Xiao , Shaodong Cao , Gianluca Setti , Zhaowen Qiu, and Xin Gao.

King Abdullah University of Science and Technology, KAUST

## Installation
To prepare the environment, execute the following commands:
```
conda create -n HorusEye python==3.10
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pydicom==2.4.4
pip install monai
```

## Predict
### Denoising Prediction
For denoising tasks, please utilize [inference.py](https://github.com/Arturia-Pendragon-Iris/HorusEye_XRT_Restoration/blob/main/inference.py) and run the "predict_denoised_slice" for 2D image prediction: .
Ensure that you replace the checkpoint path with your local checkpoint file. The inference process typically completes within a few seconds, yielding restored results.

A pretrained checkpoint (trained on approximately 1 million images) is available for code testing purposes:
[Download Pretrained Checkpoint](https://drive.google.com/file/d/1D5mhuJNszGElek5n10F8fUhvg1bA1S7f/view?usp=sharing). 

### Denoising with model-based/zero-shot methods
We provide baseline implementations for model-based and zero-shot denoising in [common_denoise.py](https://github.com/Arturia-Pendragon-Iris/HorusEye_XRT_Restoration/blob/main/common_denoise.py). 
- Example datasets are available in the ["example_dataset"](https://github.com/Arturia-Pendragon-Iris/HorusEye_XRT_Restoration/tree/main/example_dataset) folder.
- A comprehensive Colab illustration is provided here: [Colab](https://github.com/Arturia-Pendragon-Iris/HorusEye_XRT_Restoration/blob/main/Denoising_Illustration.ipynb).

### Model-based iterative reconstruction methods
We offer detailed implementations of model-based iterative reconstruction (MBIR) methods.
These methods apply the Radon transform followed by iterative MBIR techniques to achieve high-quality reconstructions.

The implementation is provided in the [projection](https://github.com/Arturia-Pendragon-Iris/HorusEye_XRT_Restoration/tree/main/projection) folder.

To employ MBIR methods, install the following dependencies:
```
conda install conda-forge::odl
conda install -c astra-toolbox -c nvidia astra-toolbox
```

### Other restoration tasks
Additional restoration tasks are available in the [restoration_task](https://github.com/Arturia-Pendragon-Iris/HorusEye_XRT_Restoration/tree/main/restoration_task) folders. Users may also develop novel restoration models upon the provided codebase.


## Datasets
### Example clean images
In the [example_dataset](https://github.com/Arturia-Pendragon-Iris/HorusEye_XRT_Restoration/tree/main/example_dataset), 50 clean CT images are provided. These images serve as validation data for evaluating denoising performance against synthesized noisy images generated with log-Poisson noise.

All images are normalized within the range [0, 1].
Example usage:

'''
img = np.load("../example_dataset/001.npy")

astra_proj_geom, astra_vol_geom, astra_proj_clean = simulate_noisy_proj_astra(img, noise=True, num_angles=270)
noisy_recon = FBP_ASTRA(astra_proj_geom, astra_vol_geom, astra_proj_clean)
'''
Here, noisy_recon denotes the synthesized noisy image.

After restoration, performance can be compared against the original image using the provided evaluation code:
'''
from analysis.evaluation import compare_img

psnr, ssim, nmse, nmae = compare_img(img, restored)
'''

### Dose-comparison dataset
You can access the dose-comparison datasets through the [link](https://drive.google.com/drive/folders/1ihSIX5sFhNzvc0Whs6dXROyCFuQTaMvM?usp=sharing). 

Hyperlinks to other public datasets are provided in Supplementary Note 1 of the Supplementary Information.

## HorusEye schematic and development
![](https://github.com/Arturia-Pendragon-Iris/HorusEye/blob/main/figures/fig_1.png)

## HorusEye on medical CT denoising
![](https://github.com/Arturia-Pendragon-Iris/HorusEye/blob/main/figures/fig_2.png)

## HorusEye on other modalities
![](https://github.com/Arturia-Pendragon-Iris/HorusEye/blob/main/figures/fig_3.png)

## HorusEye on other restoration tasks
![](https://github.com/Arturia-Pendragon-Iris/HorusEye/blob/main/figures/fig_4_1.png)

## Acknowledgement
The project is inspired by the following projects:
- [RED CNN](https://github.com/SSinyu/RED-CNN)
- [CT Former](https://github.com/wdayang/CTformer)
- [MAP NN](https://github.com/hmshan/MAP-NN)
- [RAM](https://github.com/matthieutrs/ram)
- [SGM](https://zenodo.org/records/10531170)

We highly appreciate Dr. Jinwu Zhou, Chongxinan Pet Hospital, and Anhong Pet Hospital, Hefei for providing the animal CT scans.

If you have any problem with our paper, code, or employed dataset, you can email Yuetan Chu (yuetan.chu@kaust.edu.sa), Gongning Luo (gongning.luo@kaust.edu.sa) or Xin Gao (xin.gao@kaust.edu.sa) for more information.


## Public X-ray Tomography dataset
You can download the example datasets to test our model.
- [PENET](https://github.com/marshuang80/PENet)
- [RSNA-PE](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pe-detection-challenge-2020)
- [RAD chest](https://cvit.duke.edu/resource/rad-chestct-dataset/)
- [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K)
- [DeepLesion](https://nihcc.app.box.com/v/DeepLesion)
- [Nutrient-dependent growth underpinned the Ediacaran transition to large body size](https://zenodo.org/records/4938539)
- [Revision of Icacinaceae from the Early Eocene London Clay flora based on X-ray micro-CT](https://zenodo.org/records/5022536)
- [Arm waving in stylophoran echinoderms: three-dimensional mobility analysis illuminates cornute locomotion](https://zenodo.org/records/3961994)
- [Battery Pouch Cell with Defects](https://zenodo.org/records/8189323)
- [TomoBank](https://tomobank.readthedocs.io/en/latest/)


## License
This project is covered under the Apache 2.0 License.
