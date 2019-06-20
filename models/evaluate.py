#!/usr/bin/env python3

import os
import os.path as osp
import glob
import subprocess
from argparse import ArgumentParser


def eval_fd(test_file, proto, model, compute_mode):
    base = osp.basename(model).replace('.caffemodel', '')
    detections_file = 'metrics/detections_iter_%s.xml' % base
    log_file = 'metrics/metrics_iter_%s.txt' % base
    cmd = """
          mkdir metrics
          python3 $CAFFE_ROOT/python/get_detections.py  --compute_mode {cm} --gt {test} --def {proto} --net {model} --labels "['background','face']" --resize_to 300x300 --delay -1 --det {det}
          python3 $CAFFE_ROOT/python/eval_detections.py --gt {test} --det {det} --objsize 16 1024 --imsize 1024 1024 --reasonable --mm --class_lbl face 2>&1 | tee {log}
          """.format(test=test_file, proto=proto, model=model, det=detections_file, log=log_file, cm=compute_mode)
    return cmd


def eval_pd(test_file, proto, model, compute_mode):
    base = osp.basename(model).replace('.caffemodel', '')
    detections_file = 'metrics/detections_iter_%s.xml' % base
    log_file = 'metrics/metrics_iter_%s.txt' % base
    cmd = """
            python3 $CAFFE_ROOT/python/get_detections.py --compute_mode {cm} --gt {test} --def {proto} --net {model} --labels "['background', 'person']" --resize_to 680x400 --delay -1  --det {det}
            python3 $CAFFE_ROOT/python/eval_detections.py --gt {test} --det {det} --objsize 100 10000 --imsize 1920 1080 --reasonable --mm --class_lbl person 2>&1 | tee {log}
          """.format(test=test_file, proto=proto, model=model, det=detections_file, log=log_file, cm=compute_mode)
    return cmd


def eval_ad(test_file, proto, model, compute_mode):
    log_file = 'action_metrics/iter_%s.txt' % osp.basename(model).replace('.caffemodel', '')
    cmd = """
          mkdir -p $(dirname {log})
          python3 $CAFFE_ROOT/python/action_metrics.py --compute_mode {cm} -t {test} -p {proto} -w {model} 2>&1 | tee {log}
          """.format(test=test_file, proto=proto, model=model, log=log_file, cm=compute_mode)
    return cmd


def eval_ad_event(test_file, proto, model, compute_mode):
    log_file = 'action_event_metrics/iter_%s.txt' % osp.basename(model).replace('.caffemodel', '')
    cmd = """
          mkdir -p $(dirname {log})
          python3 $CAFFE_ROOT/python/action_event_metrics.py --compute_mode {cm} -t {test} -p {proto} -w {model} 2>&1 | tee {log}
          """.format(test=test_file, proto=proto, model=model, log=log_file, cm=compute_mode)
    return cmd


def eval_cr(test_file, proto, model, compute_mode):
    base = osp.basename(model).replace('.caffemodel', '')
    detections_file = 'metrics/detections_iter_%s.json' % base
    log_file = 'metrics/metrics_iter_%s.txt' % base
    labelmap_file = '$CAFFE_ROOT/python/lmdb_utils/labelmap_cr.prototxt'
    cmd = """
          mkdir metrics
          python3 $CAFFE_ROOT/python/get_crossroad_detections.py {proto} {model} {labelmap} --compute_mode {cm} annotation {test} --annotation_out {det}
          python3 $CAFFE_ROOT/python/eval_crossroad_detections.py {test} {det} 2>&1 | tee {log}
          """.format(test=test_file, proto=proto, model=model, det=detections_file, log=log_file,
                     cm=compute_mode, labelmap=labelmap_file)
    return cmd


def eval_ag(test_file, proto, model, compute_mode):
    log_file = 'metrics/iter_%s.txt' % osp.basename(model).replace('.caffemodel', '')
    cmd = """
          mkdir -p $(dirname {log})
          python3 $CAFFE_ROOT/python/eval_age_gender.py --compute_mode {cm} --gt {test} --def {proto} --net {model} 2>&1 | tee {log}
          """.format(test=test_file, proto=proto, model=model, log=log_file, cm=compute_mode)
    return cmd


def find_files(path, iter):
    proto = 'deploy.prototxt'
    snapshots = glob.glob(osp.join(path, 'snapshots', '*_%s.caffemodel' % iter))
    if not snapshots:
        print('Snapshots from %s iteration does not exists' % iter)
        exit(1)
    assert len(snapshots) == 1
    return proto, os.path.relpath(snapshots[0], start=path)


def main():
    parser = ArgumentParser()
    parser.add_argument('--type', choices=['fd', 'pd', 'ad', 'ad_event', 'cr', 'ag'], help='Type of metric')
    parser.add_argument('--dir', required=True, help='Experiment directory')
    parser.add_argument('--iter', required=True, help='Iteration of snapshots')
    parser.add_argument('--data_dir', required=True, help='Directory with dataset')
    parser.add_argument('--annotation', required=True, help='Name of annotation file')
    parser.add_argument('--gpu', default='0', help='GPU ids')
    parser.add_argument('--image', default="ttcf:latest", help='Docker image')
    args = parser.parse_args()

    is_gpu = args.gpu != '-1'
    exec_bin = 'nvidia-docker' if is_gpu else 'docker'
    compute_mode = 'GPU' if is_gpu else 'CPU'

    container_name = 'evaluate-%s-%s' % (args.type, args.iter)

    docker_command = [
        exec_bin, 'run', '--rm',
        '--name', container_name,
        '--user=%s:%s' % (os.getuid(), os.getgid()),
        '-v', '%s:/workspace' % args.dir,
        '-v', '%s:/data:ro' % args.data_dir,  # Mount directory with dataset
        args.image
    ]

    proto, model = find_files(args.dir, args.iter)
    command = {
        'fd': eval_fd("/data/" + args.annotation, proto, model, compute_mode),
        'pd': eval_pd("/data/" + args.annotation, proto, model, compute_mode),
        'ad': eval_ad("/data/" + args.annotation, proto, model, compute_mode),
        'ad_event': eval_ad_event("/data/" + args.annotation, proto, model, compute_mode),
        'cr': eval_cr("/data/" + args.annotation, proto, model, compute_mode),
        'ag': eval_ag("/data/" + args.annotation, proto, model, compute_mode),
    }[args.type]

    try:
        subprocess.call(docker_command + ['bash', '-c', command], env={'NV_GPU': args.gpu})
    except KeyboardInterrupt:
        pass
    finally:
        subprocess.call(['docker', 'stop', '-t', '0', container_name],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print('\n'*5)
        print('='*64)
        print('Docker: %s' % ' '.join(docker_command))
        print('Command: %s' % command)


if __name__ == '__main__':
    main()
