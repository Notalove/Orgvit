import numpy as np
import cv2
import os


all_img_files = []
save_path = './data_diffdrug/'
# 遍历指定路径下的所有文件和子文件夹
for root, dirs, files in os.walk("./DiffDrug"):
    for file in files:
        if file.endswith('.tif'):
            all_img_files.append(os.path.join(root, file))

for file_path in all_img_files:
    print(file_path)
    img_metainfo = file_path.split('\\')
    print(len(img_metainfo))
    print(img_metainfo)
    if len(img_metainfo) == 4:
        drugname = img_metainfo[1]
        dayname = img_metainfo[2]
        img_filename = img_metainfo[3]
        wellname = img_filename.split('_')[0]
    elif len(img_metainfo) == 5:
        drugname = img_metainfo[1] + '-' + img_metainfo[2]
        dayname = img_metainfo[3]
        img_filename = img_metainfo[4]
        wellname = img_filename.split('_')[0]
    else:
        raise("error path")
    
    new_img_filename = drugname + '_' + dayname + '_' + wellname + '.jpg'
    img_save_path = save_path + new_img_filename

    img = cv2.imread(file_path)
    cv2.imwrite(img_save_path, img)

    
    # try:
    #     image = Image.open(file_path)
    # except:
    #     print('Open Error! Try again!')
    #     continue
    # else:
    #     r_image, cls_nums_statis = unet.detect_image(image, count=count, name_classes=name_classes)
    #     line = file_path + "," + str(cls_nums_statis[0]) + "," + str(cls_nums_statis[1]) + "," + str(cls_nums_statis[2]) + "," + str(cls_nums_statis[3]) + "\n"
    #     csv_file.write(line)