#!/usr/bin/env python3

import os
import glob
import json
from jinja2 import Template
from argparse import ArgumentParser
from collections import OrderedDict


DEFAULT_NAME = "person_detection_action_recognition_{}_classes"


def generate_variables(model_name, num_actions, fine_tune):
    num_classes = num_actions + 1  # + background

    variables = dict()
    variables['model_name'] = model_name
    variables['num_actions'] = num_actions
    variables['num_classes'] = num_classes
    variables['fine_tune'] = fine_tune
    variables['valid_actions_ids'] = str(list(range(0, num_actions)))
    variables['valid_classes_ids'] = str(list(range(0, num_classes)))
    variables['ignore_class_id'] = num_actions

    class_names_map = OrderedDict()
    valid_class_names = list()
    for i in range(0, num_actions):
        class_names_map['class_label_%s' % i] = i
        valid_class_names.append('class_label_%s' % i)

    class_names_map['__undefined__'] = num_actions
    variables['class_names_map'] = json.dumps(class_names_map, indent=4)
    variables['valid_class_names'] = json.dumps(valid_class_names)

    return variables


def render(template_file, vars, ouput_file):
    t = Template(open(template_file).read())

    dir_name = os.path.dirname(ouput_file)
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
    output_file = open(ouput_file, 'w')
    output_file.write(t.render(**vars))


def main():
    parser = ArgumentParser(description='Generator of person_detection_action_recognition model for N classes')
    parser.add_argument('--num_actions', '-n', type=int, required=True, help='Number of actions')
    parser.add_argument('--model_name', '-m', type=str, default="", help='Name of model ')
    parser.add_argument('--output_dir', '-o', type=str, default='../..', help='Directory with models')
    parser.add_argument('--fine_tune', action='store_true', help='Use solver for fine-tune with Pretrained Models')
    args = parser.parse_args()

    model_name = args.model_name if args.model_name else DEFAULT_NAME.format(args.num_actions)

    output_dir = os.path.abspath(os.path.join(args.output_dir, model_name))

    assert not os.path.exists(output_dir), "Directory already exists"

    variables = generate_variables(model_name, args.num_actions, args.fine_tune)
    file_list = glob.glob(r'templates/**/*j2', recursive=True)

    for file_name in file_list:
        base_name = file_name.replace('templates/', '')
        ouput_file = os.path.join(output_dir, base_name.replace('.j2', ''))
        render(file_name, variables, ouput_file)

    print('')
    print('Model: %s' % model_name)
    print('Output direcotry: %s' % output_dir)
    print('Next steps:')
    print(' 1. Set correct "class_names_map" and "valid_class_names" in %s/data_config.json' % output_dir)
    print(' 2. Train model: ')
    print('    $ cd <repo>/models')
    print('    $ python train.py --model %s --weights action_detection_0005.caffemodel --work_dir <WORK_DIR> --data_dir <PATH_TO_DATA> --gpu 0' % model_name)


if __name__ == '__main__':
    main()
