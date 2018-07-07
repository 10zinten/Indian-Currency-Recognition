# Indian-Currency-Recognition
A project for visually impaired people to recognise the bills of Indian National Rupee using Deep Learning technology.


### Training the model
```
python train.py \
--train_dir=../data/currency/train_dir \
--val_dir=../data/currency/test_dir \
--nb_epoch=3 \
--batch_size=32
```
