import os
import sys
import glob
import argparse
import logging
import datetime

from tensorflow.keras import __version__
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# setting up logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler('./logs/train.log')
logger.addHandler(file_handler)


logging.info("=================================================================")
logging.info(datetime.datetime.now().ctime())

IM_WIDTH, IM_HEIGHT = 224, 224  # fixed size for ResNet50
LR = 0.001
NB_EPOCHS = 5
BATCH_SIZE = 32
FC_SIZE = 1024
NB_RN50_LAYERS_TO_FREEZE = 172
base_model = ResNet50

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
    Args:Optional array of the same length as x, containing weights to ap
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
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


def setup_to_finetune(model):
    """Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.

    note: NB_IV3_LAYERS corresponds to the top2 inception blocks in the inceptionv3 arch

    Args:
        model: keras model
    """
    for layer in model.layers[:NB_RN50_LAYERS_TO_FREEZE]:
        layer.trainable = False
    for layer in model.layers[NB_RN50_LAYERS_TO_FREEZE:]:
        layer.trainable = True
    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

def plot_training(h):
    acc = h.history['acc']
    val_acc = h.history['val_acc']
    loss = h.history['loss']
    val_loss = h.history['val_loss']
    epochs = range(len(acc))
    print(val_acc)
    print(acc)
    print(loss, val_loss)

    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r-')
    plt.xlabel('num of epochs')
    plt.ylabel('accuracy')
    plt.grid(True)
    plt.legend(['Train', 'Val'])
    plt.title('Training Vs validation accuracy')
    plt.show()

    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.xlabel('num of epochs')
    plt.ylabel('loss')
    plt.grid(True)
    plt.legend(['Train', 'Val'])
    plt.title('Training Vs Validation loss')
    plt.show()

def train(args):
    
    # logging hyperparameters
    logging.info("HYPERPARAMETERS:")
    logging.info("model =ResNet50")
    logging.info("learning-rate ={}".format(args.lr))
    logging.info("nb_epoches ={}".format(args.nb_epoch))
    logging.info("batch-size ={}".format(args.batch_size))
    logging.info("nb-layer-to-freeze ={}".format(args.nb_layer_to_freeze))

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
        epochs=args.nb_epoch,
        validation_data=validation_generator,
        class_weight='auto')

    # fine-tuning
    setup_to_finetune(model)

    history_ft = model.fit_generator(
        train_generator,
        epochs=args.nb_epoch,
        validation_data=validation_generator,
        class_weight='auto')

    model.save(args.output_model_file)

    if args.plot:
        plot_training(history_tl)

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--train_dir')
    a.add_argument('--val_dir')
    a.add_argument('--lr', type=int, default=LR)
    a.add_argument('--nb_epoch', type=int, default=NB_EPOCHS)
    a.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    a.add_argument('--nb_layer_to_freeze', type=int, default=NB_RN50_LAYERS_TO_FREEZE)
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
