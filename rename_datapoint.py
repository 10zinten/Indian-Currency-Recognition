import os

PATH = '/home/tenzin/ML/project/dl-research-lab/data/dataset'

sorted_classes = sorted(os.listdir(PATH))
for id, dir_name in enumerate(sorted_classes):
    for img in os.listdir(os.path.join(PATH, dir_name)):
        source = os.path.join(PATH, dir_name, img)
        destination = os.path.join(PATH, '{}_IMG_{}'.format(id, img))
        os.rename(source, destination)
