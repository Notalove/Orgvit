import cv2
import numpy as np 
import glob

path = './data/VOCdevkit/VOC2012/SegmentationClass/*.png'
save_path = './data/VOCdevkit/VOC2012/SegmentationClass/'
for i,img_path in enumerate(glob.glob(path)):
    #读入图像
    print(img_path)
    mask = cv2.imread(img_path,0)
    print(mask.shape)
    image_filename = img_path.split('/')[-1]
    image_filename = image_filename.split('_masks_organoid')[-2] + '_img.png'
    print(image_filename)
    h, w = mask.shape[:2]
    new_img = np.zeros((h, w))
    new_img = np.ceil(np.divide(mask, 255)).astype(np.uint8)
    a,b = np.unique(new_img, return_counts=True)
    print(dict(zip(a,b)))
    print(new_img.shape)
    cv2.imwrite(save_path + image_filename, new_img)
    print(i,'-------------------------')