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
To test the pretrained mode, first download the full dataset from the [link](http://cseweb.ucsd.edu/~viscomp/projects/SIGA18ShapeSVBRDF/Data.zip), which is around 128GB. Unzip the dataset under the same directory where you save the code. And then run 
```
python test.py --cuda 
```
The testing errors will be saved in the folder `test_render2_refine1_cascade2` and painted on the screen. We have corrected a bug when rendering the second bounce image. Therefore the testing errors are slightly different from the number in the paper, which are summarized as follows.  

|      |Albedo_0|Albedo_1|Albedo_2|Normal_0|Normal_1|Normal_2|
|------|-------|-------|-------|-------|-------|-------|
|New|5.649x10<sup>-2</sup>|5.116x10<sup>-2</sup>|4.843x10<sup>-2</sup>|4.513x10<sup>-2</sup>|3.898x10<sup>-2</sup>|3.815x10<sup>-2</sup>|
|Origin|5.670x10<sup>-2</sup>|5.132x10<sup>-2</sup>|4.868x10<sup>-2</sup>|4.580x10<sup>-2</sup>|3.907x10<sup>-2</sup>|3.822x10<sup>-2</sup>|
|      |**Roughness_0**|**Roughness_1**|**Roughness_2**|**Depth_0**|**Depth_1**|**Depth_2**|
|New|2.061x10<sup>-1</sup>|2.0072x10<sup>-1</sup>|1.938x10<sup>-1</sup>|1.865x10<sup>-2</sup>|1.620x10<sup>-2</sup>|1.501x10<sup>-2</sup>| 
|Origin|2.064x10<sup>-1</sup>|2.011x10<sup>-1</sup>|1.943x10<sup>-1</sup>|1.871x10<sup>-2</sup>|1.624x10<sup>-2</sup>|1.505x10<sup>-2</sup>|
|      |**Bounce1_0**|**Bounce1_1**|**Bounce1_2**|**Bounce2_0**|**Bounce2_1**|**Bounce2_2**|
|New|3.309x10<sup>-3</sup>|2.042x10<sup>-3</sup>|1.648x10<sup>-3</sup>|2.65x10<sup>-4</sup>|2.22x10<sup>-4</sup>|2.19x10<sup>-4</sup>|
|Origin|3.291x10<sup>-3</sup>|2.046x10<sup>-3</sup>|1.637x10<sup>-3</sup>|2.76x10<sup>-4</sup>|2.47x10<sup>-4</sup>|2.45x10<sup>-4</sup>|
|      |**Bounce3_0**|**Bounce3_1**|**Bounce3_2**| | | |
|New|6.5x10<sup>-5</sup>|5.9x10<sup>-5</sup>|5.8x10<sup>-5</sup>||||
|Origin|6.4x10<sup>-5</sup>|5.9x10<sup>-5</sup>|5.8x10<sup>-5</sup>||||

## Train
