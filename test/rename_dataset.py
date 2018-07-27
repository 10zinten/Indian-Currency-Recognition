import os
from PIL import Image


data_dir = '/home/srmdlrl/project/data/currency'
new_data_dir = os.path.join(data_dir, 'currency-v0.2')

train_dir = os.path.join(data_dir, 'train_dir')
test_dir = os.path.join(data_dir, 'test_dir')


def rename(split_dir):
    
    prefix = lambda c_id, filename: "{}_IMG_{}.jpg".format(c_id, filename.split('.')[0])
    
    for idx, cls_name in enumerate(sorted(os.listdir(split_dir))):
        class_path = os.path.join(split_dir, cls_name)
        for filename in os.listdir(class_path):
            image = Image.open(os.path.join(class_path, filename))
            renamed = prefix(idx, filename)
            image.save(os.path.join(split_dir, renamed))


if __name__ == '__main__':
    # rename(train_dir)
    rename(test_dir)
        

