#!/usr/bin/env python3

import os
import os.path as osp
import shutil
import subprocess
from argparse import ArgumentParser
from datetime import datetime


def next_number_of_experiment(path):
    list = [0] + [int(x) for x in os.listdir(path) if x.isdigit()]
    next = max(list) + 1
    return str(next).rjust(3, '0')


def prepare_directory(path, model_name, is_gpu):
    if not osp.exists(path):
        print('Directory does not exist: %s' % path)
        exit(1)

    if not osp.exists(model_name):
        print('Directory does not exist: %s' % model_name)
        exit(1)

    model_dir = osp.join(path, model_name)
    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    number_of_experiment = next_number_of_experiment(model_dir)
    experiment_dir = osp.join(model_dir, number_of_experiment)

    assert not osp.exists(experiment_dir)

    shutil.copytree(model_name, experiment_dir)

    # Rewrote solver.protoxt to run in CPU mode
    if not is_gpu:
        solver_path = osp.join(experiment_dir, 'solver.prototxt')
        with open(solver_path) as file:
            new_text = file.read().replace('GPU', 'CPU')

        with open(solver_path, "w") as file:
            file.write(new_text)

    os.makedirs(osp.join(experiment_dir, 'snapshots'))
    os.makedirs(osp.join(experiment_dir, 'logs'))

    return experiment_dir, number_of_experiment


def main():
    parser = ArgumentParser()
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--weights', help='The pretrained weights from "init_weights" directory')
    parser.add_argument('--work_dir', required=True, help='Directory to collect result of training process')
    parser.add_argument('--data_dir', required=True, help='Directory with dataset')
    parser.add_argument('--solver', default='solver.prototxt', help='The solver definition protocol buffer text file.')
    parser.add_argument('--gpu', default='0', help='GPU ids')
    parser.add_argument('--image', default="ttcf:latest", help='Docker image')

    args = parser.parse_args()

    assert os.path.exists(args.work_dir), "Directory does not exist"
    assert os.path.exists(args.data_dir), "Directory does not exist"

    is_gpu = args.gpu != '-1'
    exec_bin = 'nvidia-docker' if is_gpu else 'docker'
    experiment_dir, experiment_num = prepare_directory(args.work_dir, args.model, is_gpu)

    container_name = "%s-%s" % (args.model, experiment_num)
    docker_command = [
        exec_bin, 'run', '--rm',
        '--user=%s:%s' % (os.getuid(), os.getgid()),
        '--name', container_name,  # Name of container
        '-v', '%s:/workspace' % os.path.abspath(experiment_dir),  # Mout work directory
        '-v', '%s:/data:ro' % os.path.abspath(args.data_dir),  # Mount directory with dataset
        '-v', '%s:/init_weights:ro' % os.path.abspath('../init_weights'),  # Mount directory with init weights
        args.image
    ]

    caffe_command = 'caffe train --solver=%s' % args.solver
    if is_gpu:
        caffe_command += ' --gpu=all'
    if args.weights:
        weights = [osp.join('/init_weights', x) for x in args.weights.split(',')]
        caffe_command += ' --weights="%s"' % ",".join(weights)

    caffe_log_command = ' 2>&1 | tee logs/%s.log' % datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    try:
        subprocess.call(docker_command + ['bash', '-c', caffe_command + caffe_log_command], env={'NV_GPU': args.gpu})
    except KeyboardInterrupt:
        pass
    finally:
        print('\n'*5)
        print('='*64)
        print('Directory: %s' % experiment_dir)
        print('Docker: %s' % " ".join(docker_command))
        print('Command: ' + caffe_command)
        print('='*64)
        print('\nStopping container...')
        subprocess.call(['docker', 'stop', '-t', '0', container_name],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


if __name__ == '__main__':
    main()
