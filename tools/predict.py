import argparse
import os
import os.path as osp
import mmcv
import torch
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import init_segmentor, inference_segmentor
# from mmseg.datasets import build_dataloader, build_dataset
# from mmseg.models import build_segmentor
from IPython import embed
import numpy as np
import cv2 
import csv


def parse_args():
    parser = argparse.ArgumentParser(
        description='OrganoidViT inference a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('imagefolder', help='full path of images for inferrence')
    parser.add_argument('suffix', help='suffix of image file')
    # parser.add_argument(
    #     '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    # parser.add_argument('--out', default='work_dirs/res.pkl', help='output result file in pickle format')
    # parser.add_argument(
    #     '--format-only',
    #     action='store_true',
    #     help='Format the output results without perform evaluation. It is'
    #     'useful when you want to format the result to a specific format and '
    #     'submit it to the test server')
    # parser.add_argument(
    #     '--eval',
    #     type=str,
    #     nargs='+',
    #     default='mIoU',
    #     help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
    #     ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', default='./res_tmp', help='directory where painted images will be saved')
    # parser.add_argument(
    #     '--gpu-collect',
    #     action='store_true',
    #     help='whether to use gpu to collect results.')
    # parser.add_argument(
    #     '--tmpdir',
    #     help='tmp directory used for collecting results from multiple '
    #     'workers, available when gpu_collect is not specified')
    # parser.add_argument(
    #     '--options', nargs='+', action=DictAction, help='custom options')
    # parser.add_argument(
    #     '--eval-options',
    #     nargs='+',
    #     action=DictAction,
    #     help='custom options for evaluation')
    # parser.add_argument(
    #     '--launcher',
    #     choices=['none', 'pytorch', 'slurm', 'mpi'],
    #     default='none',
    #     help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    model = init_segmentor(args.config, args.checkpoint, device='cuda:0')
    image_folder = args.imagefolder
    with open(args.show_dir + '/statis.csv', 'w', newline='', encoding='utf8') as f:
        writer = csv.writer(f)
        writer.writerow(["filename","live_num","live area","death_area","bubble_area"])
        for root, dirs, files in os.walk(image_folder):
            for file in files:
                if file.endswith(args.suffix):
                    image_file = os.path.join(root, file)
                    result = inference_segmentor(model, image_file)
                    mask_and = np.zeros(result[0].shape, dtype=np.uint8)
                    center = (result[0].shape[0]//2, result[0].shape[1]//2)
                    radius = np.min(np.array(result[0].shape)) // 2
                    print(center, radius)
                    cv2.circle(mask_and, center, radius, (255,), thickness=-1, lineType=None, shift=None)
                    # print(result[0])
                    # print(mask_and)
                    # print(result[0].dtype)
                    # print(mask_and.dtype)
                    result = cv2.bitwise_and(result[0].astype(np.uint8), mask_and)

                    #num_labels, labels, _, _ = cv2.connectedComponentsWithStats(result, connectivity=8, ltype=cv2.CV_32S)
                    live_mask = np.zeros(result.shape, dtype=np.uint8)
                    live_mask[np.where(result==1)] = 1
                    num_live, _, _, _ = cv2.connectedComponentsWithStats(live_mask, connectivity=8, ltype=cv2.CV_32S)
                    num_live = num_live - 1

                    pix_statis = {}
                    print("$$$$$$$$$$$$$$$$")
                    print(result.shape)
                    a,b = np.unique(result, return_counts=True)
                    print(dict(zip(a,b)))
                    print("$$$$$$$$$$$$$$$$")
                    pix_statis = dict(zip(a,b))
                
                    pix_live = 0
                    pix_death = 0
                    pix_bubble = 0
                    pix_bg = 0
                    for key, value in pix_statis.items():
                        if key == 0:
                            pix_bg = value
                        elif key == 1:
                            pix_live = value
                        elif key == 2:
                            pix_death = value
                        elif key == 3:
                            pix_bubble = value
                        else:
                            print("\033[1;31;40m Bad Pixel Value:\033[0m", key)
                            continue

                    if args.show:
                        img_tensor = cv2.imread(image_file)
                        img_name = file.split('.')[0]
                        #img_info_list = img_name.split('_')
                        #drug, daynum, wellnum = img_name.split('_')
                        #print(drug, daynum, wellnum)
                        # if save_mask == True:
                        #     #mask_save_name = drug + "_" + daynum + "_" + wellnum + "_mask.png"
                        #     mask_save_name = img_name
                        #     mask_save_path = osp.join(out_dir, mask_save_name)
                        #     mask_raw = result[0]
                        #     mask = mask_raw * (255 // np.max(mask_raw))
                        #     cv2.imwrite(mask_save_path, mask)
                        #     print("\033[1;32;40m save mask:\033[0m", mask_save_path)
                        sta_line = []
                        # sta_line.append(drug)
                        # sta_line.append(daynum)
                        # sta_line.append(wellnum)
                        sta_line.append(img_name)
                        sta_line.append(num_live)
                        sta_line.append(pix_live)
                        sta_line.append(pix_death)
                        sta_line.append(pix_bubble)
                      
                        
                        writer.writerow(sta_line) 

                        if args.show_dir:
                            out_file = osp.join(args.show_dir, file)
                        else:
                            out_file = None
                        
                        if args.show:
                            showimg = True
                        else:
                            showimg = False
                        
                        if hasattr(model, 'module'):
                            model = model.module
                        model.show_result(
                            img_tensor,
                            [result],
                            palette = model.PALETTE,
                            show = showimg,
                            out_file=out_file)
                    
if __name__ == "__main__":
    main()
