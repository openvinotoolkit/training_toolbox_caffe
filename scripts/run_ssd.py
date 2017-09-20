import argparse

import caffe
import cv2
import numpy as np
from caffe.proto import caffe_pb2
from google.protobuf import text_format
from tqdm import tqdm

from datasets_toolbox.common.persistence import dump
from datasets_toolbox.common.image_grabbers import ImageGrabber
from datasets_toolbox.common.drawing import VideoWriter, draw_rect, fit_image_to_window


def load_labels(label_map_file_path):
    label_map = caffe_pb2.LabelMap()
    label_map_serialized = open(label_map_file_path, 'rt').read()
    text_format.Merge(str(label_map_serialized), label_map)
    labels = [str(item.display_name) for item in label_map.item]
    return labels


def get_detection(raw_detection, original_frame_size):
    assert raw_detection.size == 7, 'Wrong number of elements in raw detector output'
    image_id = int(raw_detection[0])
    class_id = int(raw_detection[1])
    score = float(raw_detection[2])
    bbox = np.clip(raw_detection[3:7], 0, 1)
    bbox *= np.tile(original_frame_size, 2)
    bbox = bbox.astype(int)
    return image_id, class_id, score, bbox


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for running SSD detector on images/video',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('prototxt', help='Path to prototxt model description')
    parser.add_argument('caffemodel', help='Path to caffemodel binary weights file')
    parser.add_argument('labelmap', help='Path to file with label names')
    parser.add_argument('source_type', choices=('video', 'imdb', 'annotation'), default='video',
                        help='Image source type')
    parser.add_argument('source', help='Path to images source')

    parser.add_argument('--annotation_out', dest='annotation_out_file_path', default=None,
                        help='Path to annotation file to save detections to')
    parser.add_argument('-t', '--threshold', dest='confidence_threshold', default=0.05, type=float,
                        help='confidence threshold to filter out weak detections')
    parser.add_argument('--gpu', dest='gpu_id', default=0, type=int,
                        help='GPU device id to use. Negative for CPU.')
    parser.add_argument('--vis', action='store_true',
                        help='Whether to show image with detections on screen or not')
    parser.add_argument('--window_size', nargs=2, type=int, default=(768, 1024),
                        help='Maximum window size to fit output image to')
    parser.add_argument('--delay', default=0, type=int,
                        help='Delay between two frames')
    parser.add_argument('--mean', nargs=3, default=(104, 117, 123), type=float,
                        help='Pixel mean value to subtracted before feeding the image to the net')
    parser.add_argument('--video_out', default=None, type=str,
                        help='Path to video file to write resulting images to')
    parser.add_argument('--caption', action='store_true',
                        help='Whether to print caption with class labels and score on top of each detected box')
    parser.add_argument('--output_blob', dest='output_blob_name', default='detection_out',
                        help='Name of network output blob')
    args = parser.parse_args()

    if args.gpu_id < 0:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)

    net = caffe.Net(args.prototxt, caffe.TEST, weights=args.caffemodel)
    assert len(net.inputs) == 1, 'Single input blob is expected'
    input_data_shape = net.blobs[net.inputs[0]].data.shape
    assert input_data_shape[0] == 1, 'Only batch 1 is supported'
    assert input_data_shape[1] == 3, 'Color image is expected'
    input_width = input_data_shape[3]
    input_height = input_data_shape[2]

    data_provider = ImageGrabber.create_image_grabber(args.source_type, args.source)

    class_labels = load_labels(args.labelmap)
    annotation = []

    video_out = VideoWriter(args.video_out)

    for frame_src, image_name in tqdm(data_provider):
        frame = np.copy(frame_src)
        frame = frame.astype(np.float32)
        frame_height, frame_width = frame.shape[:2]
        frame = cv2.resize(frame, (input_width, input_height), interpolation=cv2.INTER_LINEAR)
        frame -= args.mean
        frame = frame.transpose((2, 0, 1))[np.newaxis, ...]

        output_blobs = net.forward(data=frame)
        assert args.output_blob_name in output_blobs, \
            'The net has multiple output blobs and none of them has name "detection_out"'
        detections = output_blobs[args.output_blob_name].reshape(-1, 7)

        annotation.append({'image': image_name,
                           'objects': []
                           })

        for detection in detections:
            image_id, class_id, score, bbox = get_detection(detection, [frame_width, frame_height])
            if score > args.confidence_threshold:
                annotation[-1]['objects'].append({'label': class_labels[class_id],
                                                  'bbox': list(bbox),
                                                  'score': score})

                if args.vis or args.video_out is not None:
                    cyan_color = (255, 255, 0)
                    caption = None
                    if args.caption:
                        caption = '{} {:.2}'.format(class_labels[class_id], score)
                    draw_rect(frame_src, rect=bbox, caption=caption, color=cyan_color)

        if args.vis or args.video_out is not None:
            frame_src = fit_image_to_window(frame_src, args.window_size)[0]
        if args.vis:
            cv2.imshow('detections', frame_src)
            if cv2.waitKey(args.delay) == 27:
                break
        video_out.append(frame_src)

    if args.annotation_out_file_path is not None:
        dump(annotation, args.annotation_out_file_path)
