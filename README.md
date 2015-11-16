# NOAA

Learning to use Theano by playing with the NOAA whale recognition data ([link](https://www.kaggle.com/c/noaa-right-whale-recognition)).

## Instructions

First, ensure all dependencies are met.  You will need at least the equivalent of:

- Anaconda (python 2.7)
- OpenCV

Next, `mv config-example.py config.py` and edit to match your local directory structure (to replicate results, leave the `SEED` variable unchanged).

Next `cd /your/base/directory/noaa/data` and run:

```bash
$ python build_head_detection_training_set.py
# change -num to match the number of samples you generated
$ opencv_createsamples -info head_examples.info -num 1000 -w 48 -h 48 -vec heads.vec
# change -num and -numNeg to match the number of samples you generated
$ opencv_traincascade -data heads -vec heads.vec -bg head_backgrounds.info \
    -numPos 1875 -numNeg 8000 -numStages 10 -w 48 -h 48 -featureType HAAR -mode ALL \
    -precalcValBufSize 2048 -precalcIdxBufSize 2048
```

Now `cd /your/base/directory/noaa/code` and run:

```bash
# additional steps
$ python preproc.py
$ python deep_convolutional_neural_network.py
```

## Notes

Status: unstable, under active development.
If using numpy 1.10.0, you may need to edit Theano's source code per [this commit](https://github.com/Theano/Theano/commit/bdcb752aa9abcaf8a7fb1e8e56d981e9bc151058).
