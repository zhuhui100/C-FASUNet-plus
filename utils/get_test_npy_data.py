import SimpleITK as sitk
import cv2
import os
import numpy as np



def get_nii_data(data_path):
    image = sitk.ReadImage(data_path)
    image_arr = sitk.GetArrayFromImage(image)
    image_arr = np.transpose(image_arr, (1, 2, 0))

    return image_arr



def save_slice(img_dir, save_dir):

    for i, name in enumerate( os.listdir(img_dir)):
        volume = get_nii_data(img_dir + name)       # [96:400, 172:396]
        volume_name = name[:-4]     # Patient_41
        # print(volume.dtype, type(volume) )   
        # if i > 0:
        #     break 

        for j in range(volume.shape[2]):
            s = str(j).zfill(3)
            np.save('%s/%s_%s.npy'%(save_dir, volume_name, s), volume[:,:,j])



def get_txt(img_dir, txt_save_path):
    nameList = []
    for filename in os.listdir(img_dir):
        nameList.append(filename)

    nameList = sorted(nameList, key=lambda x: x)

    txt = "../dataset_2D/test_41_60/txt/test_41_60.txt"

    with open(txt, 'w') as f:
        count = 0
    
        for filename in nameList:
            name = filename.split('.')

            # write image path in txt file
            f.write( img_dir + filename)
            f.write('\n')
            count += 1
    
    print('image number: ', count)



if __name__ == "__main__":

    img_dir = '../nii_41_60/'
    save_img_dir = "../dataset_2D/test_41_60/img/"
    txt_save_path = "../dataset_2D/test_41_60/txt/test_41_60.txt"

    save_slice(img_dir, save_img_dir)

    get_txt(save_img_dir, txt_save_path)

    print('finish')     