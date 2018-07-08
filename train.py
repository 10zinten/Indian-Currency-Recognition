import os
import sys
import glob
import argparse

from keras import __version__
from resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


IM_WIDTH, IM_HEIGHT = 224, 224  # fixed size for ResNet50
NB_EPOCHS = 3
BATCH_SIZE = 32
FC_SIZE = 1024
IV3_LAYERS_TO_FREEZE = 172

def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    count = 0
    for r, dirs, files in os.walk(directory):
        count += len(files) if files else 0
    return count

def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    Args:
        base_model: keras model excluding top
        nb_classes: # of classes

    Returns:
        model: keras model with custom out layer
    """
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(FC_SIZE, activation='relu')(x)  #new FC layer, random init
    x = x = Dropout(0.5)(x)
    predictions = Dense(nb_classes, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model

def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

def plot_training(h):
    acc = h.history['acc']
    val_acc = h.history['val_acc']
    loss = h.history['loss']
    val_loss = h.history['val_loss']
    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r-')
    plt.xlabel('num of epochs')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.legend(['Train', 'Val'])
    plt.title('Training Vs validation accuracy')

    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.xlabel('num of epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend(['Train', 'Val'])
    plt.title('Training Vs Validation loss')
    plt.show()

def train(args):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""
    nb_train_samples = get_nb_files(args.train_dir) - (222-50)  # samples=150
    nb_classes = len(glob.glob(args.train_dir + '/*'))
    nb_val_samples = get_nb_files(args.val_dir)
    nb_epoch = int(args.nb_epoch)
    batch_size = int(args.batch_size)

    # data preparation
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size
    )

    validation_generator = test_datagen.flow_from_directory(
        args.val_dir,
        target_size=(IM_WIDTH, IM_HEIGHT),
        batch_size=batch_size
    )

    # setup model
    base_model = ResNet50(weights='imagenet', include_top=False) # Excludes final FC layer
    model = add_new_last_layer(base_model, nb_classes)

    # transfer learning
    setup_to_transfer_learn(model, base_model)

    history_tl = model.fit_generator(
        train_generator,
        nb_epoch=nb_epoch,
        samples_per_epoch=nb_train_samples,
        validation_data=validation_generator,
        nb_val_samples=nb_val_samples,
        class_weight='auto')

    model.save(args.output_model_file)

    if args.plot:
        plot_training(history_tl)

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--train_dir')
    a.add_argument('--val_dir')
    a.add_argument('--nb_epoch', default=NB_EPOCHS)
    a.add_argument('--batch_size', default=BATCH_SIZE)
    a.add_argument('--output_model_file', default='inceptionv3-ft.model')
    a.add_argument('--plot', action='store_true')

    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print('directories do not exist')
        sys.exit(1)

    train(args)
