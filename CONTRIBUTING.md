# Contributing

## Merge Requests

Making an MR please follow a few simple yet important conventions:

* Clearly state the purpose of your MR. E.g.: enable support of some network architecture, or propose a fix for a bug with a clear explanation what the bug is.
* If your aim is to enable some ad-hoc architecture, put a link to the model (prototxt and caffemodel) itself. The link to the paper describing the net would be useful as well.
* If you copy any layer from a third-party project, add a link to that project.
* If you add an implementation of some layer please first do your best to check if a layer with a similar logic does already exist in caffe-shared, and think about reusing it rather than making a slightly different copy.
* Bring minimal required code chunks to this repository. I.e. if you copy an implementation of some layer from a third-party project don't copy all other changes from that project to caffe-shared repository.

Original BVLC Caffe contribution guide can be found [here](CONTRIBUTING_BVLC.md).
