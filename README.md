# Orgvit
Semantic segmentation of organoid medical images which can identify live and dead cells
# Prediction
python tools/predict.py logs3_512_4cls/2DSegFormer.b1.512x512.aero.80k.py *.pth ./Bro tif --show --show-dir=./work_dirs
# Requirements
mmcv-full == 1.2.7
pytorch == 1.7.1+cu101
opencv-python == 4.11.0.86
