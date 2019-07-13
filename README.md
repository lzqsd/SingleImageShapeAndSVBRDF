# Learning to Reconstruct Shape and Spatially-Varying Reflectance from a Single Image 
Zhengqin Li, Zexiang Xu, Ravi Ramamoorthi, Kalyan Sunkavalli and Manmohan Chandraker. SIGGRAPH ASIA, 2018

## Quick Start
This is the official code release of paper [Learning to Reconstruct Shape and Spatially-Varying Reflectance from a Single Image ](https://drive.google.com/file/d/17K3RrWQ48gQynOhZHq1g5sQgjLjoMiPk/view). To test all the real images included in our paper, supplementary material and video, run
```
python testReal.py --cuda 
```
The input images are included in folder `real`. Their names are listed in file `imList.txt`, which will be loaded when running `testReal.py`. The results will be by default saved in folder `output`. Definitions of outputs are as follows. `(n)` is the name of input and `(m)` is the level of cascade. Since we use three levels of cascade, the value of `(m)` can be 0, 1 or 2.
* `(n)albedo_(m).png`:
* `(n)normal_(m).png`:
* `(n)rough_(m).png`:
* `(n)depth_(m).png/hdf5`:
* `(n)shCoef_(m).hdf5`:
* `(n)renderedBounce1_(m).png`:
* `(n)renderedBounce2_(m).png`:
* `(n)renderedBounce3_(m).png`:
* `(n)renderedEnv_(m).png`:

## Test

## Train
