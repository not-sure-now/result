import os
import sys

import shutil

import yaml


def install(requirements_path):
    if os.system(f'pip install -r {requirements_path}') != 0:
        return -1
    return 0


def copy(from_, to_):
    if not os.path.isfile(from_):
        print(f'{from_} not found!')
        exit(1)
    shutil.copyfile(from_, to_)
    return 0


def main():
    # restrict input
    if len(sys.argv) > 2:
        print('too many input!')
        exit(1)

    if len(sys.argv) <= 1:
        print('please input cityscapes root!')
        exit(1)

    # get cityscapes root
    origin_root = sys.argv[1]
    if not os.path.isdir(origin_root):
        print(f'{origin_root} is not a dir!')
        exit(1)

    # get project root dir
    root = os.path.dirname(__file__)
    if root == "":
        root = '.'

    # get requirement file
    requirement = f'{root}/requirements.txt'
    if not os.path.isfile(requirement):
        print('requirement file missing!')
        exit(1)

    # install prerequisite
    if install(requirement) != 0:
        print('requirement install failure!')
        # exit(1)

    # prepare dataset
    # if os.system(f'python cityscapes2yolo.py {origin_root}') != 0:
        # exit(1)

    # try to update the ultralytics to run my model and add intel extension
    try:
        import ultralytics
    except ImportError:
        print('ultralytics not found!')
        exit(1)
    ultralytics_root = os.path.dirname(ultralytics.__file__)

    # modify ultralytics to enable training

    # overwrite default.yaml(this will allow me to add pruning and distillation option to cfg file)
    if copy(f'{root}/overwrite/default.yaml', f'{ultralytics_root}/cfg/default.yaml') != 0:
        print('copy error!')
        exit(1)

    # overwrite nn.modules.block.py to add my block
    if copy(f'{root}/overwrite/block.py', f'{ultralytics_root}/nn/modules/block.py') != 0:
        print('copy error!')
        exit(1)
    # overwrite nn.modules.__init__.py to enable export of my block
    if copy(f'{root}/overwrite/__init__.py', f'{ultralytics_root}/nn/modules/__init__.py') != 0:
        print('copy error!')
        exit(1)

    # overwrite trainer.py to enable pruning and distillation
    if copy(f'{root}/overwrite/trainer.py', f'{ultralytics_root}/engine/trainer.py') != 0:
        print('copy error!')
        exit(1)

    # overwrite trainer.py to enable pruning and distillation
    if copy(f'{root}/overwrite/__init__1.py', f'{ultralytics_root}/cfg/__init__.py') != 0:
        print('copy error!')
        exit(1)

    # overwrite tasks.py to enable parsing of my block and enable distillation loss hook
    if copy(f'{root}/overwrite/tasks.py', f'{ultralytics_root}/nn/tasks.py') != 0:
        print('copy error!')
        exit(1)

    datas = yaml.load(open(root + '/train.yaml').read(), Loader=yaml.FullLoader)
    datas['data'] = root + '/yolo_format/cityscapes/train/images'
    datas['model'] = root + '/yolov8.yaml'

    f = open(root + '/train.yaml', 'w+')
    f.write(yaml.dump(datas))
    f.close()








if __name__ == '__main__':
    main()






