#!/usr/bin/env python3

import glob
import os
import os.path as osp
import subprocess
from argparse import ArgumentParser

MO_BIN = '/opt/intel/openvino/deployment_tools/model_optimizer/mo_caffe.py'

def find_files(path, iter):
    proto = 'deploy.prototxt'
    snapshots = glob.glob(osp.join(path, 'snapshots', '*_%s.caffemodel' % iter))
    if not snapshots:
        print('Snapshots from %s iteration does not exists' % iter)
        exit(1)
    assert len(snapshots) == 1
    return proto, os.path.relpath(snapshots[0], start=path)


def shell_command(proto, model, data_type, output_dir, model_name):
    cmd = """
          {bin} --input_proto {proto} --input_model {model} --data_type {type} \
          --output_dir {dir} --model_name {name}""".format(
              bin=MO_BIN, proto=proto, model=model, type=data_type, dir=output_dir, name=model_name)
    return cmd


def shell_command_cr(proto, model, data_type, output_dir, model_name):
    cmd = shell_command(proto, model, data_type, output_dir, model_name) + " --mean_values [104,117,123]"
    return cmd


def shell_command_ad(model, data_type, output_dir, model_name):
    iter_path = osp.dirname(output_dir)
    intermediate_path = osp.join(iter_path, 'intermediate_models')

    model_stage1 = osp.join(intermediate_path, 'ie_stage1.caffemodel')
    proto_stage1 = '/workspace/intermediate_models/ie_conversion.prototxt'
    proto_stage2 = '/workspace/intermediate_models/inference.prototxt'

    cmd = """
          mkdir -p {intermediate_path}
          python3 /opt/caffe/python/convert_to_ie_compatible.py -m {proto} -w {model} -o {out_file}""".format(
              intermediate_path=intermediate_path, output_dir=output_dir,
              proto=proto_stage1, model=model, out_file=model_stage1)

    cmd += shell_command(proto_stage2, model_stage1, data_type, output_dir, model_name)

    return cmd


def shell_command_ag(proto, model, data_type, output_dir, model_name):
    cmd = shell_command(proto, model, data_type, output_dir, model_name) + " --scale 255.0"
    return cmd


def main():
    parser = ArgumentParser()
    parser.add_argument('--type', default='simple', choices=['simple', 'ad', 'cr', 'ag'], help='Model type')
    parser.add_argument('--dir', required=True, help='Experiment directory')
    parser.add_argument('--iter', required=True, help='Iteration of snapshots')
    parser.add_argument('--name', required=True, help='Model name')
    parser.add_argument('--data_type', default='FP32', choices=['FP16', 'FP32'],
                        help='Data type for all intermediate tensors and weights.')
    parser.add_argument('--gpu', default='0', help='GPU ids')
    parser.add_argument('--image', default="ttcf:latest", help='Docker image')
    args = parser.parse_args()

    proto, model = find_files(args.dir, args.iter)

    is_gpu = args.gpu != '-1'
    exec_bin = 'nvidia-docker' if is_gpu else 'docker'

    docker_command = [
        exec_bin, 'run', '--rm',
        '--user=%s:%s' % (os.getuid(), os.getgid()),
        '-v', '%s:/workspace' % os.path.abspath(args.dir),
        args.image
    ]

    output_dir = osp.join('mo', 'iter_'+args.iter, args.data_type.lower())
    model_name = args.name

    command = {
        'simple': shell_command(proto, model, args.data_type, output_dir, model_name),
        'ad': shell_command_ad(model, args.data_type, output_dir, model_name),
        'cr': shell_command_cr(proto, model, args.data_type, output_dir, model_name),
        'ag': shell_command_ag(proto, model, args.data_type, output_dir, model_name),
    }[args.type]

    subprocess.call(docker_command + ['bash', '-c', command], env={'NV_GPU': args.gpu})
    print('='*64)
    print('Docker: %s' % ' '.join(docker_command))
    print('Command: %s' % command)
    print('Output directory: %s' % osp.join(args.dir, output_dir))


if __name__ == '__main__':
    main()
