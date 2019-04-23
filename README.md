Caffe and MatCovNet implementations 
## Joint Demosaicing and Denoising using Deep Mean-Shift Priors

This code contain an implementation for image demosaicing based on the paper:

### Deep Mean-Shift Priors for Image Restoration ([project page](http://home.inf.unibe.ch/~bigdeli/DMSPrior.html))

Siavash Arjomand Bigdeli, Meiguang Jin, Paolo Favaro, Matthias Zwicker

Advances in Neural Information Processing Systems (NIPS), 2017

#### Abstract:
In this paper we introduce a natural image prior that directly represents a Gaussian-smoothed version of the natural image distribution. We include our prior in a formulation of image restoration as a Bayes estimator that also allows us to solve noise-blind image restoration problems. We show that the gradient of our prior corresponds to the mean-shift vector on the natural image distribution. In addition, we learn the mean-shift vector field using denoising autoencoders, and use it in a gradient descent approach to perform Bayes risk minimization. We demonstrate competitive results for noise-blind deblurring, super-resolution, and demosaicing.


<img src="http://home.inf.unibe.ch/~bigdeli/img/DMSPrior.jpg" alt="Drawing" style="height: 500px;" align="center"/>

See [manuscript](https://arxiv.org/pdf/1709.03749) for details of the method.


This code runs in Matlab and you need to install either
[MatCaffe](http://caffe.berkeleyvision.org) or
[MatConvNet](http://www.vlfeat.org/matconvnet/).
### Contents:

[demo_demosaicing.m](https://github.com/siavashBigdeli/DMSP/blob/master/demo.m): Includes an example for joint image demosaicing.

[DMSPDemosaic.m](https://github.com/siavashBigdeli/DMSP/blob/master/DMSPDeblur.m): Implements SGD for demosaicing. Use Matlab's help function to learn about the input and output arguments.

[DAEs](https://github.com/siavashBigdeli/DMSP/tree/master/DAEs): Includes DAE models and function handles (in Caffe and matconvnet).

[data](https://github.com/siavashBigdeli/DMSP/tree/master/data): Includes sample images taken from the [Microsoft Demosaicing Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=52535).

