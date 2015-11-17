# NOAA

Learning to use Theano by playing with the NOAA whale recognition data ([link](https://www.kaggle.com/c/noaa-right-whale-recognition)).

## Prerequisites

Ensure all dependencies are met.  You will need at least the equivalent of:

- Anaconda (python 2.7)
- OpenCV

Download this repository and run `mv config-example.py config.py`.   Edit `config.py` to match your local directory structure (to replicate results, leave the `SEED` variable unchanged).

## OpenCV

`cd /your/base/directory/noaa/` and run:

```bash
$ python -c 'from noaa.cascade.generate_opencv_training_data import main; main()'
$ cd data
# in the following command, change -numPos to match the number of samples you generated
$ opencv_createsamples -info head_examples.info -num 2000 -w 48 -h 48 -vec heads.vec
# in the following command, change -numPos and -numNeg to match the number of samples you generated
$ opencv_traincascade -data heads -vec heads.vec -bg head_backgrounds.info \
    -numPos 1700 -numNeg 8000 -numStages 10 -w 48 -h 48 -featureType HAAR -mode ALL \
    -precalcValBufSize 2048 -precalcIdxBufSize 2048
```

Note that `-numPos` in `opencv_traincascade` should be smaller than `-num` from `opencv_createsamples`.  According to [this thread](http://code.opencv.org/issues/1834), the relationship is `num >= (numPos + (numStages-1) * (1 - minHitRate) * numPos) + S`, where `S` is the count of examples identified as background images during training.  Another comment on the thread suggests using `numPos = .85 * num`.

## Additional steps

...

## NNet

The current nnet does not use pre-processing other than shrinking the images.  To shrink the images, first edit `proc-imgs.sh` to match your directory structure.  Then `cd /your/base/directory/noaa/` and run `source noaa/proc-imgs.sh`.

Next, load the processed data into memmap arrays.  `cd /your/base/directory/noaa/` and run:

```bash
$ python -c 'from noaa.nnet.preproc import build_memmap_arrays; build_memmap_arrays()'
```

To train the nnet, first `cd /your/base/directory/noaa/`, open the python interpreter, and run the following commands:

```python
>>> from noaa.nnet.deep_convolutional_neural_network import fit_lenet
>>> ln = fit_lenet()'
```

## Notes

Status: unstable, under active development.
If using numpy 1.10.0, you may need to edit Theano's source code per [this commit](https://github.com/Theano/Theano/commit/bdcb752aa9abcaf8a7fb1e8e56d981e9bc151058).
