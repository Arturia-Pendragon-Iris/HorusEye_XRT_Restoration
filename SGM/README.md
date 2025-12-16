# SGM for Medical Imaging
Wavelet-based Score-based Generative Model for Medical Imaging
         

## Abstract
The score-based generative model (SGM) has demonstrated remarkable performance in addressing challenging under-determined inverse problems in medical imaging.   However, acquiring high-quality training datasets for these models remains a formidable task, especially in medical image reconstructions.   Prevalent noise perturbations or artifacts in low-dose Computed Tomography (CT)or under-sampled Magnetic Resonance Imaging (MRI) hinder the accurate estimation of data distribution gradients, thereby compromising the overall performance of SGMs when trained with these data.   To alleviate this issue, we propose a wavelet-based denoising technique to cooperate with the SGMs, ensuring effective and stable training.   Specifically, the proposed method integrates a wavelet sub-network and the standard SGM sub-network into a unified framework, effectively alleviating inaccurate distribution of the data distribution gradient} and enhancing the overall stability.   The mutual feedback mechanism between the wavelet sub-network and the SGM sub-network empowers the neural network to learn accurate scores even when handling noisy samples.   This combination results in a framework that exhibits superior stability during the learning process, ultimately leading to the generation of more precise and reliable reconstructed images.   During the reconstruction process, we further enhance the robustness and quality of the reconstructed images by incorporating compressed sensing regularization.   Our experiments, which encompass various scenarios of low-dose and sparse-view CT, as well as MRI with varying under-sampling rates and masks, consistently demonstrate the effectiveness of the proposed method by significantly enhanced the quality of the reconstructed images.   Especially, our method with noisy training samples achieves comparable results to those obtained using clean data.
 

## File introduction
separate_ImageNet.py is the main file. Place the downloaded pre-trained checkpoint in the run/logs/AapmCT_10C directory, as well as the config file.
We provide an example of running a low-dose CT called testcase1.IMA, the results of which are saved in mat format at NCSN_train/L506_rec_5e3.
The runners folder contains the training and test files of the model. The configs folder contains the training parameter Settings file for the model; The main models of the network are in the Models folder; The model loss function is designed under the losses folder.


## Requirements and Dependencies
See requirement.txt for detailed environment Settings    
## Test
    python separate_ImageNet.py --model ncsn --runner Aapm_Runner_CTtest_10_noconv --config aapm_10C.yml --doc AapmCT_10C --test --image_folder output

## Checkpoints
Pre-trained checkpoints will be released soon.

##  Connection
If you have any questions about the code, please feel free to leave a comment on GitHub or contact me at mail:wuweiw7@mail.sysu.edu.cn. I will reply in due course.


