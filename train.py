import os
import sys
import glob
import argparse

from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.image import ImageDataGenerator

IM_WIDTH, IM_HEIGHT = 299, 299  # fixed size for InceptioinV3
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


def train(args):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""
    nb_train_samples = get_nb_files(args.train_dir)
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
        zoom_range=0.2,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
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

if __name__ == '__main__':
    a = argparse.ArgumentParser()
    a.add_argument('--train_dir')
    a.add_argument('--val_dir')
    a.add_argument('--nb_epoch', default=NB_EPOCHS)
    a.add_argument('--batch_size', default=BATCH_SIZE)

    args = a.parse_args()
    if args.train_dir is None or args.val_dir is None:
        a.print_help()
        sys.exit(1)

    if (not os.path.exists(args.train_dir)) or (not os.path.exists(args.val_dir)):
        print('directories do not exist')
        sys.exit(1)

    train(args)
