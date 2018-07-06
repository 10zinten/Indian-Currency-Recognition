from pathlib import Path
import random


PATH = Path('../data/currency')
TRAIN = 'train_dir'
TEST = 'test_dir'

test_split = 0.2
test_size = 0

def split_data():
    while test_size != test_split*400:
        idx = random.randint(1, 401)
        img = str(idx) + '.jpg'
        for cls_path in (PATH/TRAIN).iterdir():
            img_path = cls_path/img
            if img_path.exists():
                cls_name = str(img_path.parent).split('/')[-1]
                print(cls_name)
                if not (PATH/TEST/cls_name).exists():
                    (PATH/TEST/cls_name).mkdir(parents=True)
                img_path.replace((PATH/TEST/cls_name/img))
                test_size += 1
