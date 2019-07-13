# Learning to Reconstruct Shape and Spatially-Varying Reflectance from a Single Image 
Zhengqin Li, Zexiang Xu, Ravi Ramamoorthi, Kalyan Sunkavalli and Manmohan Chandraker. SIGGRAPH ASIA, 2018

## Real Examples
This is the official code release of paper [Learning to Reconstruct Shape and Spatially-Varying Reflectance from a Single Image ](https://drive.google.com/file/d/17K3RrWQ48gQynOhZHq1g5sQgjLjoMiPk/view). To test all the real images included in our paper, supplementary material and video, run
```
python testReal.py --cuda 
```
The input images are included in folder `real`. Their names are listed in file `imList.txt`, which will be loaded when running `testReal.py`. Results are by default saved in folder `output`. Definitions of results are as follows. `(n)` is the name of input and `(m)` is the level of cascade. Since we use three levels of cascade, the value of `(m)` can be 0, 1 or 2.
* `(n)albedo_(m).png`: Diffuse albedo prediction without gama correction. 
* `(n)normal_(m).png`: Normal prediction.
* `(n)rough_(m).png`: Roughness prediction.
* `(n)depth_(m).png/hdf5`: `.hdf5` is real depth. `.png` is tranformed depth in range of 0 to 1.  
* `(n)shCoef_(m).hdf5`: Spherical harmonics coefficents prediction for environment lighting. 
* `(n)renderedBounce1_(m).png`: First bounce image rendered using predicted shape and SVBRDF. 
* `(n)renderedBounce2_(m).png`: Second bounce image predicted by global illumination network. 
* `(n)renderedBounce3_(m).png`: Third bounce image predicted by global illumination network.
* `(n)renderedEnv_(m).png`: Image rendered using predicted environment lighting, shape and SVBRDF. 
* `(n)input.png`: Input image. 
* `(n)mask.png`: Segmentation mask. 

## Test
To test the pretrained mode, first download the dataset from the [link](http://cseweb.ucsd.edu/~viscomp/projects/SIGA18ShapeSVBRDF).

## Train
