# HorusEye
This is the official repository of "HorusEye: A self-supervised foundation model for generalizable X-ray tomography restoration"

## Installation
```
conda create -n HorusEye python==3.10
pip install pydicom==2.4.4
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## HorusEye schematic and performance evaluation on dose-comparison data
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

We highly appreciate Dr. Jinwu Zhou, Chongxinan Pet Hospital, and Anhong Pet Hospital, Hefei for providing the animal CT scans.

## Public dataset for training and testing
The public datasets used in this study are publicly available and can be accessed via their respective websites as follows.
- [PENET](https://github.com/marshuang80/PENet)
- [RSNA-PE](https://www.rsna.org/rsnai/ai-image-challenge/rsna-pe-detection-challenge-2020)
- [RAD chest](https://cvit.duke.edu/resource/rad-chestct-dataset/)
- [Medical Segmentation Decathlon (MSD)](http://medicaldecathlon.com/)
- [AbdomenCT-1K](https://github.com/JunMa11/AbdomenCT-1K)
- [CHAOS dataset](https://chaos.grand-challenge.org/)
- [DeepLesion](https://nihcc.app.box.com/v/DeepLesion)
- [AAPM LDCT](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=52758026)
- [Verse Dataset](https://github.com/MIRACLE-Center/CTPelvic1K)
- [CTPelvic1K](https://github.com/MIRACLE-Center/CTPelvic1K)
- [RSNA Intracranial Hemorrhage](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection/data)
- [CQ500](http://headctstudy.qure.ai/dataset)
- [Piglet dataset](https://github.com/xinario/SAGAN)
- [Luna 16](https://luna16.grand-challenge.org/)
- [MIDRC dataset](https://www.rsna.org/covid-19/covid-19-ricord)
- [KiTS 19](https://github.com/neheller/kits19)
- [TCIA Colonography dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=3539213)
- [TCIA HCC-TACE dataset](https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=70230229)
- [SpineWeb dataset](http://spineweb.digitalimaginggroup.ca/)
- [CTooth dataset](https://github.com/liangjiubujiu/CTooth)
