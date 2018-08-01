import argparse
import random
import os

from PIL import Image
from tqdm import tqdm

SIZE = 128

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../data/dataset', help='Directory with the currency dataset')
parser.add_argument('--output_dir', default='../data/224x224_currency', help='where to write the new dataset')


def resize_and_save(filename, output_dir, size=SIZE):
    """Resize the image contianed in `filename` and save it to the `output_dir`"""
    image = Image.open(filename)
    # use bilinear interpolation instead of the default "nearest neighbor" method
    image = image.resize((size, size), Image.BILINEAR)
    image.save(os.path.join(output_dir, filename.split('/')[-1]))

if __name__ == '__main__':
    args = parser.parse_args()

    assert os.path.isdir(args.data_dir), "Couldn't find the dataset at {}".format(args.data_dir)

    # define the data directories
    train_data_dir = os.path.join(args.data_dir, 'train_dir')
    dev_data_dir = os.path.join(args.data_dir, 'dev_dir')
    test_data_dir = os.path.join(args.data_dir, 'test_dir')

    # Get the filenames in dataset directory
    filenames = os.listdir(args.data_dir)
    filenames = [os.path.join(args.data_dir, f) for f in filenames if f.endswith('.jpg')]

    # split the images in 'currenct' into 80% train, 10% dev 10% test
    # make sure to always shuffle with a fixed seed so that spllit is reproducible
    random.seed(230)
    filenames.sort()
    random.shuffle(filenames)

    split_1 = int(0.8 * len(filenames))
    split_2 = int(0.9 * len(filenames))
    train_filenames = filenames[:split_1]
    dev_filenames = filenames[split_1: split_2]
    test_filenames = filenames[split_2:]

    filenames = {'train': train_filenames,
                 'dev': dev_filenames,
                 'test': test_filenames}

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print('Warning: output dir {} already exists'.format(args.output_dir))

    # Preprocess train, dev and test
    for split in ['train', 'dev', 'test']:
        output_dir_split = os.path.join(args.output_dir, '{}_dir'.format(split))
        if not os.path.exists(output_dir_split):
            os.mkdir(output_dir_split)
        else:
            print('Warning: dir {} already exits'.format(output_dir_split))

        print('Processing {} data, saving preprocessed data to {}'.format(split, output_dir_split))
        for filename in tqdm(filenames[split]):
            resize_and_save(filename, output_dir_split, size=SIZE)

    print("Done building dataset")
