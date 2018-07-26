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
--output_model_file=resnet50-tf-l5.h5 \
--nb_layer_to_freeze=5
```

## Project sturcture based on [CS230 Deep Learning](https://cs230-stanford.github.io/project-code-examples.html)
```
data/
    train/
    dev/
    test/
experiments/
    base_model/
        params.json
        ...
    learning_rate/
        lr_0.1/
            params.json
        lr_0.01/
            params.json
    batch_norm/
        params.json
model/
    input_fn.py
    model_fn.py
    utils.py
    training.py
    evaluation.py
build_dataset.py
train.py
search_hyperparams.py
synthesize_results.py
evaluate.py
```
Here is each file or directory’s purpose:
- ```data/```: contain all the data of the project, with an explicit train/dev/test split.
- ```experiments```: contains the different experiments (like base_model, learning_rate, batch_norm). Each directory (experiment) after training will contain multiple things:
  - ```params.json```: the list of hyperparameters, in json format
  - ```train.log```: the training log (everything print to the console)
  - ```train_summaries```: train summaries for TensorBoard
  - ```eval_summaries```: eval summaries for TensorBoard
  - ```last_weights```: weights saved from the 5 last epoches
  - ```best_weights```: best wieghts (based on dev accuracy)
- ```model/```: module defining the model and function used in train or eval.
  - ```input_fn.py```: where we define the input data pipeline
  - ```model_fn.py```: creates the deep learning model
  - ```utils.py```: utilitu function for handling hyperparams / logging
  - ```training.py```: utility functions to train a model
  - ```evaluation.py```: utility functions to evaluate a model
- ```build_dataset.py```: creates or trainsforms the dataset, built the split into train/dev/test in reproducible way.
- ```train.py```: train the model on input data, and evaluate each epoch on the dev set
- ```search_hyperparams.py```: run ```train.py``` multiple times with different hyperparameters
- ```synthesize_result.py```: expolre different experiments in a directory and display a nice table of the results
- ```evaluate.py```: evaluate the model on the test set (should be run once at the end of the project

### Refrences
- [How to use transfer learning and fine-tuning in Keras and Tensorflow to build an image recognition system and classify (almost) any object](https://deeplearningsandbox.com/how-to-use-transfer-learning-and-fine-tuning-in-keras-and-tensorflow-to-build-an-image-recognition-94b0b02444f2)
- [Transfer Learning: retraining Inception V3 for custom image classification](https://becominghuman.ai/transfer-learning-retraining-inception-v3-for-custom-image-classification-2820f653c557)
- [How to Retrain an Image Classifier for New Categories](https://www.tensorflow.org/hub/tutorials/image_retraining)
