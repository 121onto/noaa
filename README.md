# NOAA

Learning to use Theano by playing with the NOAA whale recognition data ([link](https://www.kaggle.com/c/noaa-right-whale-recognition)).

## Instructions

First, ensure all dependencies are met.  You will need at least the equivalent of:

- Anaconda (python 2.7)
- OpenCV

Next, `mv config-example.py config.py` and edit to match your local directory structure (to replicate results, leave the `SEED` variable unchanged).

Finally, `cd /your/base/directory/noaa/code` and run:

```bash
$ python build_head_detection_training_set.py
$ opencv_createsamples -info ../data/head_examples.info -num 4545 -w 256 -h 256 -vec heads.vec
$ opencv_traincascade -data . -vec heads.vec -bg head_backgrounds.info -numPos 4545 -numNeg 36360 -numStages 2 -w 256 -h 256 -featureType HAAR
...
# additional steps

$ python preproc.py
$ python deep_convolutional_neural_network.py
```

## Notes

Status: unstable, under active development.
If using numpy 1.10.0, you may need to edit Theano's source code per [this commit](https://github.com/Theano/Theano/commit/bdcb752aa9abcaf8a7fb1e8e56d981e9bc151058).
