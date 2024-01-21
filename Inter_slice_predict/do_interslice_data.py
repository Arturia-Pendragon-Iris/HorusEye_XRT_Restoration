import os
import numpy as np

raw_path = "/data/chest_CT/rescaled_ct/xwzc"
# artery_path = "/home/chuy/Artery_Vein_Upsampling/mask/artery_1"
save_path = "/chest"
noise_path = "/xwzc_filter"
noise_list = os.listdir(noise_path)

for filename in np.sort(os.listdir(raw_path)):
    print(filename)
    raw_file = np.load(os.path.join(raw_path, filename))["arr_0"]
    raw_file = np.clip((raw_file + 1000) / 1600, 0, 1)
    np_array = np.transpose(raw_file, (2, 0, 1))
    for i in range(150, 400):
        module_input = np_array[i - 1:i + 2].copy()
        k = np.random.rand()
        if k < 0.5:
            module_input[0] += np.random.normal(size=[512, 512]) * 0.05
            module_input[2] += np.random.normal(size=[512, 512]) * 0.05
        else:
            index_1 = np.random.randint(len(noise_list))
            index_2 = np.random.randint(len(noise_list))
            module_input[0] += 1 * np.load(os.path.join(noise_path, noise_list[index_1]))["arr_0"]
            module_input[2] += 1 * np.load(os.path.join(noise_path, noise_list[index_2]))["arr_0"]

        number = str(i).zfill(3)
        save_name = os.path.join(save_path, number + filename.replace(".npy", ".npz"))
        np.savez_compressed(save_name, module_input)
