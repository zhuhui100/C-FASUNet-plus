## Segmentation of Organs at Risk Based on a Cascaded FAS-UNet+ Framework with Iterative Optimization Strategy

This repository provides the Pytorch code for "Segmentation of Organs at Risk Based on a Cascaded FAS-UNet+ Framework with Iterative Optimization Strategy".

### Requirements
Some important required packages include:
* Python == 3.8.15 
* Pytorch == 1.13.1 
* Some basic Python packages, such as Numpy, skimage.

## Usages
### For test datasets
1. First, you can download the dataset at [SegTHOR challenge][data link] and put them in './nii_41_60'. To save the dataset as ".npy", run:

[data_link]:https://codalab.lisn.upsaclay.fr/competitions/843#learn_the_details
```
cd ./utils/
python3 get_test_npy_data.py
```

2. Put pth files. 

    (1). Download the 'pth' file of coarse and fine segmentation models from [[BaiduPan [code:code]][coarse models link], [[BaiduPan [code:code]][fine models link].

    [coarse models link]: https://pan.baidu.com/s/1eYl4AIvT2KbqHOFGkwOMdA
    [fine models link]: https://pan.baidu.com/s/129P1Kdb8c0bRHmfZJP3ZTg

    (2). Place coarse ('123_s3_fold*_1e6_avg_best.pth') and fine ('123_fold*_net2_s3_net1_avg_best.pth') segmentation models in  './test/pth_net_1/' and './test/pth_net_2_by_net1/'  directory, respectively.   
    
    The files format is as follows:
```
    /test/pth_net_1/
        123_s3_fold1_1e6_avg_best.pth
        ...
        23_s3_fold5_1e6_avg_best.npy
    /test/pth_net_2_by_net1/
        123_fold1_net2_s3_net1_avg_best.pth
        ...
        123_fold5_net2_s3_net1_avg_best.pth
```
    
3. To test cascaded FAS-UNet+ on test dataset, run:

```
cd ./test/
python3 two_stage_weight_pred.py
```

  The segmentation results are saved in the './test/2stage_result/' directory.

