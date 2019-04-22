# Data preparation

**NOTE** Only for models that used `custom_layer/data_layer.py`

Assume next structure of data:
<pre>
    |-- data_dir
         |-- videos
            video1.mp4
            video2.mp4
            video3.mp4
         |-- images
            |-- video_1
                frame_000000.png
                frame_000001.png
            |-- video_2
                frame_000000.png
                frame_000001.png
         |-- annotation
            annotation_file_1.xml
            annotation_file_2.xml
            annotation_file_3.xml
         train_tasks.txt
         test_tasks.txt
</pre>
Each annotation file (see [this](#annotation-file-format) header) describes a single source of images (see [this](#image-file-format) header).

## Annotation file format
For annotating it's better to use [CVAT](https://github.com/opencv/cvat) utility. So we assume that annotation file is stored in appropriate `.xml` [format](https://github.com/opencv/cvat/blob/develop/cvat/apps/documentation/xml_format.md). In annotation file we have single independent track for each person on video which includes of bounding box description on each frame. General structure of annotation file:
<pre>
    |-- root
         |-- track_0
              bounding_box_0
              bounding_box_1
         |-- track_1
              bounding_box_0
              bounding_box_1
</pre>
 Toy example of annotation file:
```xml
<?xml version="1.0" encoding="utf-8"?>
<annotations count="1">
    <track id="0" label="person">
        <box frame="0" xtl="1.0" ytl="1.0" xbr="0.0" ybr="0.0" occluded="0">
            <attribute name="action">action_name</attribute>
        </box>
    </track>
</annotations>
```
where fields have next description:
 - `count` - number of tracks
 - `id` - unique ID of track in file
 - `label` - label of track (data loader will skips all other labels except `person`)
 - `frame` - unique ID of frame in track
 - `xtl`, `ytl`, `xbr`, `ybr` - bounding box coordinates of top-left and bottom-right corners
 - `occluded` - marker to highlight heavy occluded bounding boxes (can be skipped during training)
 - `name` - name of bounding box attribute (data loader is sensitive for `action` class only)
 - `action_name` - valid name of action (you can define own list of actions)

## Image file format
Our implementation of data loader works with independent images stored on the drive. Each image should be named in format `frame_xxxxxx.png` or `frame_xxxxxx.jpg` (where `xxxxxx` is unique image number).

**NOTE** To extract images from video you can use `caffe/tools/extract_images.py`

## Train tasks file format
For more robust control of image sources we have created separate file where each row represents a single source in next format: `annotation_file_path.xml image_height,image_width images_directory_path`. We assume that all images from the same source are resize to `image_height,image_width` sizes (it needs to properly decode annotations).

Example of `train_tasks.txt` file:
```
annotations/annotation_file_1.xml 1920,1080 images/video1
annotations/annotation_file_2.xml 1920,1080 images/video2
```

Example of `test_tasks.txt` file:
```
annotations/annotation_file_3.xml videos/video3.mp4
```
train_tasks.txt
