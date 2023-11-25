import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import sys
sys.path.append("../")


import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as data
import  torch.nn.functional as F

import numpy as np
import datetime
import SimpleITK as sitk
import cv2
import shutil
import monai


import argparse
import warnings
warnings.filterwarnings('ignore')


from fasunet_plus_2d import FAS_Unet_plus_2d             


class MyDataset_test(data.Dataset):
    """

    """

    def __init__(self, data_root_dir, txt_path, shuffle=False):
        with open(txt_path, 'r') as f:
            data_path = [x.strip("\n") for x in f.readlines()]
        data_path.sort()
        
        
        self.imgList = [data_root_dir + x.split(",")[0] for x in data_path]
    
    def __getitem__(self, index):
        
        img = np.load(self.imgList[index])[96:400, 172:396]
        img_src = np.load(self.imgList[index])


        # to tensor
        img_src = np.expand_dims(img_src, axis=0)           
        img_src = torch.from_numpy(img_src).float()
        
        img = np.expand_dims(img, axis=0)           
        img = torch.from_numpy(img).float()


        name = self.imgList[index].split("/")[-1]

        return (img, name, img_src)
    
    def __len__(self):
        return len(self.imgList)





def save_to_nii_512(volume_npy, name, save_dir):

    pred_3D = volume_npy
    save_path = save_dir + name + '.nii'

    nii_arr = np.array(pred_3D).astype('uint8')
    
    sitk_img = sitk.GetImageFromArray(nii_arr, isVector=False)
    sitk.WriteImage(sitk_img, save_path)

    # img = nib.Nifti1Image(nii_arr, np.eye(4))
    # nib.save(img, save_path)
    print('file save path:{}, '.format(save_path), 
          'shape: ', nii_arr.shape,
         'max:', nii_arr.max(), 
          'min:', nii_arr.min(),
         )



def connected_component(image, thr=1):
  
    from skimage import measure

    label, num = measure.label(image, connectivity=1, return_num=True)
    if num < 1:
        return image

    region = measure.regionprops(label)

    
    num_list = [i for i in range(1, num+1)]
    area_list = [region[i-1].area for i in num_list]  
    num_list_sorted = sorted(num_list, key=lambda x: area_list[x-1])[::-1]

    # rei=move
    if len(num_list_sorted) > thr:  
        for i in num_list_sorted[thr:]:
            label[region[i-1].slice][region[i-1].image] = 0
    num_list_sorted = num_list_sorted[:thr]

    label[label > 0] = 1   
    return label


    
def get_img_roi(img, pred):
    '''
    img: 512x512 numpy array
    pred: 512x512 numpy array
    '''
    
    crop = monai.transforms.CropForegroundd(keys=['image', 'label'],
                                source_key='label',
                                start_coord_key='foreground_start_coord', 
                                end_coord_key='foreground_end_coord'

                                )
    
    max_x1, max_x2 = 136, 360
    max_y1, max_y2 = 188, 380
    
    if pred.max() > 0:
        
        max_h, max_w = 224, 192
        

        img_copy = np.expand_dims(img.copy(), 0)
        pred_copy  = np.expand_dims(pred.copy(), 0)

        data_dicts = {"image": img_copy, "label": pred_copy}
        data_dicts = crop(data_dicts)
            
        h = data_dicts['foreground_end_coord'][0] - data_dicts['foreground_start_coord'][0]
        w = data_dicts['foreground_end_coord'][1] - data_dicts['foreground_start_coord'][1]
        dh = (max_h - h)/2
        dw = (max_w - w)/2
        if (max_h - h)%2==0:
            x_1 =  int(data_dicts['foreground_start_coord'][0] - dh )
            x_2 = int(data_dicts['foreground_end_coord'][0] + dh)
        else:
            x_1 =  int(data_dicts['foreground_start_coord'][0] - (dh +0.5))
            x_2 = int(data_dicts['foreground_end_coord'][0] + (dh-0.5) )

        if (max_w - w)%2==0:
            y_1 =  int(data_dicts['foreground_start_coord'][1] - dw)
            y_2 = int(data_dicts['foreground_end_coord'][1] + dw)
        else:
            y_1 =  int(data_dicts['foreground_start_coord'][1] - (dw+0.5))
            y_2 = int(data_dicts['foreground_end_coord'][1] + (dw-0.5) )

        img = img[x_1:x_2, y_1:y_2]
        pred = pred[x_1:x_2, y_1:y_2]
        roi_x = [x_1, x_2]
        roi_y = [y_1, y_2]
        # print(img.shape, pred.shape, data_dicts['foreground_start_coord'],
        #       data_dicts['foreground_end_coord'], ' crop result')
        
    else:
        img = img[max_x1:max_x2, max_y1:max_y2]
        pred = pred[max_x1:max_x2, max_y1:max_y2]
        roi_x = [max_x1, max_x2]
        roi_y = [max_y1, max_y2]
        
    img = np.expand_dims(np.expand_dims(img, axis=0), axis=0)
    img = torch.from_numpy(img).float()
    
    # pred = torch.from_numpy(np.array(pred, dtype=np.int32)).long()
    pred = np.expand_dims(np.array(pred, dtype=np.int32), axis=0)
    
    # print(img.shape, pred.shape, 'roi shape')
    
    return img, pred, roi_x, roi_y


def post_process_fun(pred_3D, post_class):
    '''
        post_class: list
    '''
    post_result = np.zeros_like(pred_3D)
    all_class = [1, 2, 3, 4]
    res_class = list(set(all_class)-set(post_class))
    
    for i in post_class:
        mask = (pred_3D == i).astype(int)
        mask = connected_component(1 - mask)
        mask = connected_component(1 - mask)
        post_result = post_result + mask * i
        print('post-processing')
        
    for i in res_class:
        post_result = (pred_3D == i).astype(int) * i + post_result

    if post_result.max() > 4 :
        print('error...................')
        raise RuntimeError('value error ')
    pred_3D = post_result.astype('uint8')
    
    return pred_3D
    

def weght_ensemble(pred_1, pred_2):
    # the result of third classes by ored_1
    
    pred = np.zeros_like(pred_1)
    
    pred = pred * (pred_2!=1).astype(np.int) + (pred_2==1).astype(np.int) * 1
    pred = pred * (pred_2!=2).astype(np.int) + (pred_2==2).astype(np.int) * 2
    pred = pred * (pred_2!=4).astype(np.int) + (pred_2==4).astype(np.int) * 4
     
    pred = pred * (pred_1!=3).astype(np.int) + (pred_1==3).astype(np.int) * 3
    if pred.max()>4:
        print('value error')
    
    return pred.astype('uint8')

    
def two_stage_avg_predict(args, net_1_list, net_2_list, device, data_loader, post=False):
    
    num_model = len(net_1_list)
    pred_list = []
    net_1_pred_list = []
    
    last_name = None
    

    
    with torch.no_grad():
        for num, (x, name, x_src) in enumerate(data_loader):
        
            inputs = x.to(device)
            name = name[0][0:10]        # Patient_41

           
            if last_name is not None and last_name != name:
                pred_2_3D = np.array(pred_list).astype('uint8')[:,0,:,:]
                pred_1_3D = np.array(net_1_pred_list).astype('uint8')[:,0,:,:]
                print(pred_1_3D.shape, pred_2_3D.shape, '---------')
                
                pred_3D = weght_ensemble(pred_1_3D, pred_2_3D)
                pred_3D = post_process_fun(pred_3D, post_class=[2, 4])
                
                save_to_nii_512(pred_3D, last_name, args.save_dir)
                pred_list.clear()
                net_1_pred_list.clear()
                
            ########## coarse seg
            for k in range(1, num_model+1):
                logits = net_1_list[k-1](inputs)
                globals()['prob_'+str(k)] = F.softmax(logits, dim=1)[0]
            
            ################## ensemble :average
            avg_prob_1 = torch.zeros_like(prob_1).to(device)  # (5, 304, 224)
            for k in range(1, num_model+1):
                avg_prob_1 +=  (globals()['prob_'+str(k)])
            avg_pred_1 = torch.argmax(avg_prob_1, dim=0).cpu().detach().numpy()
            
            
            ############ get ROI  
            net_1_pred_512 = np.zeros((1, 512, 512))
            net_1_pred_512[:, 96:400, 172:396] = avg_pred_1
                
            inputs_src = x_src[0,0].detach().numpy()
            inputs_2, target_2, roi_x, roi_y = get_img_roi(inputs_src.copy(), net_1_pred_512[0].copy())
            inputs_2 = inputs_2.to(device)
            
            
            ############# fine seg            
            for k in range(1, num_model+1):
                logits = net_2_list[k-1](inputs_2)
                globals()['prob_2_'+str(k)] = F.softmax(logits, dim=1)[0]
            
            

            ################## ensemble of fine segmentation by average
            avg_prob_2 = torch.zeros_like(prob_2_1).to(device)  # (5, 224, 192)
            for k in range(1, num_model+1):
                avg_prob_2 +=  (globals()['prob_2_'+str(k)])
            avg_pred_2 = torch.argmax(avg_prob_2, dim=0).cpu().detach().numpy()
           
            ## result restory to 512x512 in fine segmentation 
            net_2_pred_512 = np.zeros((1, 512, 512))
            # print(roi_x[0],roi_x[1], roi_y[0], roi_y[1], avg_pred_2.shape)
            net_2_pred_512[:, roi_x[0]:roi_x[1], roi_y[0]:roi_y[1]] = avg_pred_2
            
            
            pred_list.append(net_2_pred_512)
            net_1_pred_list.append(net_1_pred_512)
            

            last_name = name

        # last one
        pred_2_3D = np.array(pred_list).astype('uint8')[:,0,:,:]
        pred_1_3D = np.array(net_1_pred_list).astype('uint8')[:,0,:,:]
        print(pred_1_3D.shape, pred_2_3D.shape, '---------')

        pred_3D = weght_ensemble(pred_1_3D, pred_2_3D)
        pred_3D = post_process_fun(pred_3D, post_class=[2, 4])

        save_to_nii_512(pred_3D, last_name, args.save_dir)
        pred_list.clear()
        net_1_pred_list.clear()


        
def model_init(model_path):
    
    input_c = 1
    numc = 64
    ivs = [[3, 3, 3, 3], [7], [4, 4, 4, 4]]
    
    model = FAS_Unet_plus_2d(in_channel = 1, 
                        channels = [64,64, 64, 64,64],
                        num_classes = args.num_classes, 
                        iter_num = [[3, 3, 3, 3], [7], [4, 4, 4, 4]]
                        ).to(device)

    model.eval()
    model.load_state_dict(torch.load(model_path))
    
    return model



if __name__ == "__main__":
    
    device = "cuda"

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-classes", 
                        type=int, 
                        help="include background",
                        default=5)

    
    parser.add_argument("--net_1_pth_path", 
                        type=list, 
                        default = [
                            "./pth_net_1/123_s3_fold1_1e6_avg_best.pth",
                            "./pth_net_1/123_s3_fold2_1e6_avg_best.pth",
                            "./pth_net_1/123_s3_fold3_1e6_avg_best.pth",
                            "./pth_net_1/123_s3_fold4_1e6_avg_best.pth",
                            "./pth_net_1/123_s3_fold5_1e6_avg_best.pth",
                        ]
                       )
                            
    # best result model
    parser.add_argument("--net_2_pth_path", 
                        type=list, 
                        default = [
                            "./pth_net_2_by_net1/123_fold1_net2_s3_net1_avg_best.pth",
                            "./pth_net_2_by_net1/123_fold2_net2_s3_net1_avg_best.pth",
                            "./pth_net_2_by_net1/123_fold3_net2_s3_net1_avg_best.pth",
                            "./pth_net_2_by_net1/123_fold4_net2_s3_net1_avg_best.pth",
                            "./pth_net_2_by_net1/123_fold5_net2_s3_net1_avg_best.pth",
                        ]
                       )
    
    
    parser.add_argument("--data_root", 
                        type=str, 
                        default = "../dataset_2D/"
                        ) 
    parser.add_argument("--txt", 
                        type=str, 
                        default="../dataset_2D/test_41_60/txt/test_41_60.txt",
                        )
    parser.add_argument("--save_dir", 
                        type=str, 
                        default = "./2stage_result/") 
    

    args = parser.parse_args()


    dataset = MyDataset_test(args.data_root,
                            args.txt,
                            shuffle = False,
                            )
    data_loader = DataLoader(dataset, 
                            batch_size=1, 
                            shuffle=False, 
                            num_workers= 1 )
    

    if os.path.exists(args.save_dir):
        shutil.rmtree(args.save_dir)
    os.makedirs(args.save_dir)
    
    num_model = len(args.net_1_pth_path)
    net_1_list = []
    for i in range(0, num_model):
        net_1_list.append(model_init(args.net_1_pth_path[i]))
    
    net_2_list = []
    for i in range(0, num_model):
        net_2_list.append(model_init(args.net_2_pth_path[i]))


    print('begin prediction for test datasets')
    start_time = datetime.datetime.now()
    
    two_stage_avg_predict(args, 
            net_1_list, 
            net_2_list,
            device, 
            data_loader, 
            post = True
            )
    
    end_time = datetime.datetime.now()
    run_time = end_time - start_time
    
    print('\ntime 1: ', start_time.strftime("%d日%H:%M:%S"), 
          '\ntime 2: ', end_time.strftime("%d日%H:%M:%S"), 
          '\nrun time:', run_time
         )

    print('finish')
