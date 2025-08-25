import glob
import cv2
import numpy as np


mask_path = './test_res/*.jpg'
orin_path = './data/VOCdevkit/VOC2012/JPEGImages/'
save_path = './draw2/'
for i,img_path in enumerate(glob.glob(mask_path)):
    print(img_path)
    img_filename = img_path.split('\\')[-1]
    orin_filepath = orin_path + img_filename
    save_filepath = save_path + img_filename
    mask = cv2.imread(img_path,0)
    a,b = np.unique(mask, return_counts=True)
    print(dict(zip(a,b)))
    break
    orin = cv2.imread(orin_filepath)
    print(orin.shape)
    print(img_path,orin_filepath)
    new = mask + orin
    cv2.imwrite(save_filepath, new)
    # #读入图像
    # image = cv2.imread(img_path)
    # image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    # #读入标签
    # label_path = img_path.replace('JPEGImages','SegmentationClass')
    # label_path = label_path.replace('jpg','png')
    # label = cv2.imread(label_path,flags=0)
    # #保存npz
    # np.savez(path2+str(i),image=image,label=label)
    print('------------',i)