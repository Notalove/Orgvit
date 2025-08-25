# Orgvit
Semantic segmentation of organoid medical images which can identify live and dead cells
# Predeiction
python tools/predict.py logs3_512_4cls/2DSegFormer.b1.512x512.aero.80k.py *.pth ./Bro tif --show --show-dir=./wintest
