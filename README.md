# HorusEye for generalizable X-ray tomography restoration
This is the official repository of "HorusEye: A self-supervised foundation model for generalizable X-ray tomography restoration" by Yuetan Chu , Longxi Zhou , Gongning Luo , Kai Kang , Suyu Dong , Zhongyi Han , Lianming Wu , Xianglin Meng , Changchun Yang , Xin Guo , Yuan Cheng , Yuan Qi , Xin Liu , Dexuan Xie , Ricardo Henao , Anthony Capone , Xigang Xiao , Shaodong Cao , Gianluca Setti , Zhaowen Qiu, and Xin Gao.

King Abdullah University of Science and Technology, KAUST

## Installation
```
conda create -n HorusEye python==3.10
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pydicom==2.4.4
pip install monai
```

## Predict
### Denoising prediction
To predict your data, you can run [inference.py](https://github.com/Arturia-Pendragon-Iris/HorusEye_XRT_Restoration/blob/main/inference.py) by giving a normalized 2D image (predict_denoised_slice) or image volume (predict_denoised_volume) and replacing the checkpoint path with your local path. The program will run for several seconds and output the restored results. 

We provide our previously pretrained checkpoint with only the base dataset (about 1 million images) for the code testing. You can download the checkpoint through the [link](https://drive.google.com/file/d/1nZdp0McRwQNY6W7lE-6uRZhMQdEaPzF0/view?usp=sharing).

You can also use the [analysis/evaluation.py](https://github.com/Arturia-Pendragon-Iris/HorusEye_XRT_Restoration/blob/main/analysis/evaluation.py) to reproduce the quantitative results presented in our manuscript, including PSNR, SSIM, and FSC. 

## Dose-comparison dataset
You can access the dose-comparison datasets through the [ling](https://drive.google.com/drive/folders/1xhjMX4S019yLYYAHNuB7Q2oQFUv7Ratg?usp=sharing). The hyperlinks of other public datasets are provided in the Supplementary Note 1 presented in our Supplementary Information.

### Other restoration tasks
We provide detailed programs within the restoration_task folders. You can develop your own restoration models based on our provided codes.  


## HorusEye schematic and development
![](https://github.com/Arturia-Pendragon-Iris/HorusEye/blob/main/figures/fig_1_2.png)

## HorusEye on medical CT denoising
![](https://github.com/Arturia-Pendragon-Iris/HorusEye/blob/main/figures/fig_2_1.png)

## HorusEye on other modalities
![](https://github.com/Arturia-Pendragon-Iris/HorusEye/blob/main/figures/fig_5_1.png)

## HorusEye on other restoration tasks
![](https://github.com/Arturia-Pendragon-Iris/HorusEye/blob/main/figures/fig_4_1.png)

## Acknowledgement
The project is inspired by the following projects:
- [RED CNN](https://github.com/SSinyu/RED-CNN)
- [CT Former](https://github.com/wdayang/CTformer)
- [MAP NN](https://github.com/hmshan/MAP-NN)
- [k-SVD](https://github.com/Deepayan137/K-svd)
- [BM3D](https://github.com/Ryanshuai/BM3D_py)

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
