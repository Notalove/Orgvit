import cv2  
import numpy as np  
import glob
def rotate_image(image, angle):  
    # 获取图片尺寸  
    (h, w) = image.shape[:2]  
      
    # 旋转中心设为图片中心  
    center = (w / 2, h / 2)  
      
    # 获取旋转矩阵  
    M = cv2.getRotationMatrix2D(center, angle, 1.0)  
      
    # 执行仿射变换  
    rotated = cv2.warpAffine(image, M, (w, h))  
      
    return rotated  

def rotate_bound_white_bg(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    # -angle位置参数为角度参数负值表示顺时针旋转; 1.0位置参数scale是调整尺寸比例（图像缩放参数），建议0.75
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    # borderValue 缺失背景填充色彩，此处为白色，可自定义
    return cv2.warpAffine(image, M, (nW, nH),borderValue=(0,0,0))
    # borderValue 缺省，默认是黑色（0, 0 , 0）
    # return cv2.warpAffine(image, M, (nW, nH))


path = r'./data_diffdrug/VOCdevkit/VOC2012/JPEGImages/*.jpg'
#项目中存放训练所用的npz文件路径
path2 = r'./data_diffdrug/VOCdevkit/VOC2012/SegmentationClass/'
for i,img_path in enumerate(glob.glob(path)):
    #读入图像
    image = cv2.imread(img_path,0)
    print(image.shape)
    image_filename = img_path.split('/')[-1]
    rotated_image = rotate_bound_white_bg(image, 45)
    h, w = rotated_image.shape[:2]
    new_img = np.zeros((h//2, w//2))
    new_img = rotated_image[h//4:3*h//4, w//4:3*w//4]
    print(new_img.shape)
    # mask = np.zeros((image.shape[0], image.shape[1]))
    #cv2.imwrite(img_path, new_img)
    cv2.imwrite(path2+image_filename.replace('jpg','png'), mask)
    print(i)

    

# 加载npz文件
# data = np.load(r'G:\dataset\Unet\Swin-Unet-ori\data\Synapse\train_npz\0.npz', allow_pickle=True)
# image, label = data['image'], data['label']
print('ok')


# # 读取图片  
# image = cv2.imread('./withDeath_VOC_Rot/JPEGImages/D6_Y46-1_WellB6_Channel2X-BF_Seq0004_RGB_C1.jpg')
# img2 =   cv2.imread('./withDeath_VOC_Rot/SegmentationClass/D6_Y46-1_WellB6_Channel2X-BF_Seq0004_RGB_C1.png')
# img3 = cv2.imread('/media/lyjslay/SteamLib/Organ-Unet/code/2DSegFormer-main/data/VOCdevkit/VOC2012/SegmentationClass/D6_Y46-1_WellB6_Channel2X-BF_Seq0004_RGB_C1.png')
# img4 = cv2.imread('/media/lyjslay/SteamLib/Organ-Unet/code/2DSegFormer-main/data/VOCdevkit/VOC2012/JPEGImages/D6_Y46-1_WellB6_Channel2X-BF_Seq0004_RGB_C1.jpg')
# print(image.shape,img2.shape,img3.shape,img4.shape)
# # 旋转图片，例如旋转45度  
# rotated_image = rotate_bound_white_bg(image, 45)
# print(rotated_image.shape)  
# h, w = rotated_image.shape[:2]
# new_img = np.zeros((h//2, w//2, rotated_image.shape[2]))   
# print(new_img.shape)
# new_img = rotated_image[h//4:3*h//4, w//4:3*w//4, :]
# # 显示旋转后的图片  
# cv2.imshow('Rotated Image', new_img)  

# cv2.waitKey(0)  
# cv2.destroyAllWindows()  
  
# # 如果需要保存旋转后的图片  
# cv2.imwrite('rotated_image.jpg', rotated_image)
# cv2.imwrite('new_image.jpg', new_img)