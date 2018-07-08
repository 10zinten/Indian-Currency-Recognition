# Indian-Currency-Recognition
A project for visually impaired people to recognise the bills of Indian National Rupee using Deep Learning technology.

### Dataset:
```
currency\
  ten\
  twenty\
  fifty\
  hundrad\
  fivehundrad\
  thousand\
```
dataset can be downloaded from [here](https://drive.google.com/file/d/0B7Am6-nOVeP7N1lQOUVUYlBuc0E/edit)

### Training the model
```
python train.py \
--train_dir=../data/currency/train_dir \
--val_dir=../data/currency/test_dir \
--nb_epoch=3 \
--batch_size=32 \
--batch_size=32 \
--output_model_file=inceptionv3-tl.model
```
