# HorusEye
This is the official repository of "HorusEye: Computed Tomography Denoising via Noise Learning"

## Installation
```
conda create -n HorusEye python==3.10
pip install pydicom==2.4.4
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## HorusEye schematic and performance evaluation on dose-comparison data
![](https://github.com/Arturia-Pendragon-Iris/HorusEye/blob/main/figures/fig1-3.png)

## Examples of CT denoising
![](https://github.com/Arturia-Pendragon-Iris/HorusEye/blob/main/figures/sfig3.png)

## Clinical evaluation
![](https://github.com/Arturia-Pendragon-Iris/HorusEye/blob/main/figures/fig5-1.png)

## Temporary GUI for HorusEye denoising presentation
Here we implement a temporary GUI for HorusEye. 
```
https://huggingface.co/spaces/Altoia/HorusEye
```
Feel free to upload a .dcm file with 512×512 size and see the denoising results. Here we provide three visualization CT windows, as lung window ([-1000, 600]), the mediastinal window ([-200, 300]), and the abdomen window ([-160, 240]).

Note: The GUI is established based on the KAUST Ibex server. If there is any problem with the GUI due to the disconnection with the Ibex server, please email yuetan.chu@kaust.edu.sa. We will try to reconnect to the server or open a new address for the GUI.

## Acknowledgement
The project is inspired by the following projects:
- [RED CNN](https://github.com/SSinyu/RED-CNN)
- [CT Former](https://github.com/wdayang/CTformer)
- [MAP NN](https://github.com/hmshan/MAP-NN)

We highly appreciate Dr. Jinwu Zhou and CHongxinan Pet Hospital, Hefei for providing the animal CT scans. We are grateful to the editors and the reviewers for their time and efforts spent on our paper. Their comments are very valuable for us to improve this work. We also thank the Computational Biological Research Center of KAUST for supporting the computational resources to run the experiments. We also thank Ana Bigio, scientific illustrator for helping us with the figure illustration.

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
  
If you want to share more diverse CT scans, especially the micro-CT scans, you can find me through email at yuetan.chu@kaust.edu.sa
