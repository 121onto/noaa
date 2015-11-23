# Histograms of Oriented Gradients for Human Detection

Histograms of oriented gradients are computed over a grid of spatial regions.  For better invariance to illumination, showdowing, etc., it is useful to contrast-normalize local responses before usnig them.  This can be done by accumulating a measure of local histogram energy over somewhat larger spatial regions and using the results to normlaize all cells in the block.  Figure 1 depicts both color and gamma normalization.  Resources:

- [Image entropy using scikit image](http://scikit-image.org/docs/dev/auto_examples/plot_entropy.html)
- [Gamma correction using OpenCV](http://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/)
- [Definition of entropy via stack exchange](http://dsp.stackexchange.com/questions/15179/entropy-of-an-image)
- [Entropy of both grayscale and color image using OpenCV](https://github.com/samidalati/OpenCV-Entropy)
- [CLAHE in python using OpenCV](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#clahe-contrast-limited-adaptive-histogram-equalization)

