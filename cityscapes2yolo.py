import os
import sys

import yaml
from tqdm import tqdm

import torchvision

root = os.path.dirname(__file__)
RES_DIR_ROOT = f'{root}/yolo_format/cityscapes/'
SPLIT_DIR = ['train/', 'test/', 'val/']
IMG_DIR = 'images/'
ANN_DIR = 'labels/'

small_label = {
    'rider': 'person',
    'truck': 'car',
    'bus': 'car',
    'caravan': 'car'
}


def main():
    # restrict input
    if len(sys.argv) > 2:
        print('too many input!')
        exit(1)

    if len(sys.argv) <= 1:
        print('too few input!')
        exit(1)

    # get cityscapes root
    origin_root = sys.argv[1]
    if not os.path.isdir(origin_root):
        print(f'{origin_root} is not a dir!')
        exit(1)

    # ensure res pos
    if not os.path.isdir(f'{root}/yolo_format'):
        os.mkdir(f'{root}/yolo_format')
    if not os.path.isdir(RES_DIR_ROOT):
        os.mkdir(RES_DIR_ROOT)
    for split_dir in SPLIT_DIR:
        res_split_dir = RES_DIR_ROOT + split_dir
        label_dir = res_split_dir + ANN_DIR
        image_dir = res_split_dir + IMG_DIR

        if not os.path.isdir(res_split_dir):
            os.mkdir(res_split_dir)
        if not os.path.isdir(label_dir):
            os.mkdir(label_dir)
        if not os.path.isdir(image_dir):
            os.mkdir(image_dir)

    if os.path.isfile(root+'/cityscapes.yaml'):
        datas = yaml.load(open(root+'/cityscapes.yaml').read(), Loader=yaml.FullLoader)
        labels = datas['names']
        datas['train'] = RES_DIR_ROOT + SPLIT_DIR[0] + 'images'
        datas['val'] = RES_DIR_ROOT + SPLIT_DIR[2] + 'images'
        datas['test'] = RES_DIR_ROOT + SPLIT_DIR[1] + 'images'
        f = open(RES_DIR_ROOT + '/cityscapes.yaml', 'w+')
        f.write(yaml.dump(datas))
        f.close()
        label2idx = {labels[key]: key for key in labels.keys()}
    else:
        print('prepare the yaml please!')
        exit(1)

    # use torchvision dataset to read the dataset
    dataset_train = torchvision.datasets.Cityscapes(
        root=origin_root,
        split='train',
        mode='fine',
        target_type='polygon'
    )

    dataset_test = torchvision.datasets.Cityscapes(
        root=origin_root,
        split='test',
        mode='fine',
        target_type='polygon'
    )

    dataset_val = torchvision.datasets.Cityscapes(
        root=origin_root,
        split='val',
        mode='fine',
        target_type='polygon'
    )

    # handle each dataset
    for split_dir in SPLIT_DIR:
        print(f'{split_dir} :')
        for i, (img, label) in tqdm(enumerate(locals()[f'dataset_{split_dir[:-1]}'])):
            img.save(RES_DIR_ROOT + split_dir + IMG_DIR + f'{i}.png')
            h, w, objs = label['imgHeight'], label['imgWidth'], label['objects']
            txt = ''
            for obj in objs:
                # get correct label
                if small_label.get(obj['label'], None) is not None:
                    label = small_label[obj['label']]
                else:
                    label = obj['label']

                if label2idx.get(label, None) is None:
                    continue
                else:
                    # get the bbox using the polygon
                    x_min = w
                    y_min = h
                    x_max = 0
                    y_max = 0
                    for point in obj['polygon']:
                        x_min = min(x_min, point[0])
                        x_max = max(x_max, point[0])
                        y_min = min(y_min, point[1])
                        y_max = max(y_max, point[1])

                    x = (0.0+x_min+x_max)/w
                    y = (0.0+y_min+y_max)/h
                    h_ = (0.0+y_max-y_min)/h
                    w_ = (0.0+x_max-x_min)/w
                    txt += f'{label2idx.get(label)} {x} {y} {w_} {h_}\n'
            f = open(RES_DIR_ROOT + split_dir + ANN_DIR + f'{i}.txt', 'w+')
            f.write(txt)
            f.close()


if __name__ == "__main__":
    main()
