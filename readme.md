# Introduction

This is the Master thesis project by Dominic Spata under the tutelage of the Real-Time Computer Vision Research Group at the Institute of Neural Computation, Ruhr University Bochum.

It deals with cycle-consistent generative adversarial networks (CycleGANs), as originally developed by [Zhu et al.](https://arxiv.org/pdf/1703.10593.pdf)

The method has been used for generation of life-like traffic sign images and has been presented at the Intelligent Vehicles Symposium 2019 in Paris:

Dominic Spata, Daniela Horn, Sebastian Houben, "Generation of Natural Traffic Sign Images Using Domain Translation with Cycle-Consistent Generative Adversarial Networks", accepted for publication at the Intelligent Vehicles Symposium (IV) 2019

```
@inproceedings{SpataHornHouben2019,
	author		=	{Spata, Dominic and Horn, Daniela and Houben, Sebastian},
	title		=	{Generation of Natural Traffic Sign Images Using Domain Translation with Cycle-Consistent Generative Adversarial Networks},
	booktitle	=	{Proceedings of the IEEE Intelligent Vehicles Symposium (IV), accepted},
	year		=	{2019},
}
```

# Installation and Use

## Dependencies

The code components of this project are written for and require Python 3.6 and Tensorflow r1.10 (or higher respective versions). Some of the tools found under "code/tools" require OpenCV. The SVM-based classifier under "code/classifier/svm" requires Scikit-Learn.

- Install Python from the [official website](https://www.python.org/downloads/).
- Install Tensorflow via
```
pip3 install tensorflow
```
or
```
pip3 install tensorflow-gpu
```
Note that the latter option further requires CUDA and cuDNN (see [Tensorflow installation instructions](https://www.tensorflow.org/install) for more information).
- (Optional) Install OpenCV via
```
pip3 install opencv-python
```
- (Optional) Install Scikit-Learn via
```
pip3 install scikit-learn
```

## Using the Software

After downloading the repository, its "code" directory must be made available to the python installation by adding it to the PYTHONPATH environment variable, moving its contents into your Python installation's "Lib" folder, or adding the path manually to "sys.path" before running any scripts.

Afterwards, the applications in "code/tools", "code/cyclegan/apps", and "code/classifier" can be executed. 

Training and/or testing of a CycleGAN model requires an appropriately defined configuration file. New configurations can be created using the "code/cyclegan/apps/create_configuration.py" application. Alternatively, you can adapt the directory paths of one of the existing configurations in the "configs" folder using the "code/cylcegan/apps/customize_config.py" application. Note that the datasets these configurations refer to are not distributed alongside this project and must be acquired separately.

A minimal example for the "horse2zebra" dataset (available, among others, from [Berkeley EECS](https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/)) may look like this:

```
python -m cyclegan.apps.customize_config configs/horse2zebra_config.json configs/horse2zebra_config_custom.json
python -m cyclegan.apps.train configs/horse2zebra_config_custom.json
python -m cyclegan.apps.test configs/horse2zebra_config_custom.json
```

## Data Formats

The CycleGAN software supports all image formats supported by [tf.image.decode_image](https://www.tensorflow.org/api_docs/python/tf/image/decode_image). Those of the tools that read images as well as the SVM classifier support all image formats supported by [cv2.imread](https://docs.opencv.org/3.0-beta/modules/imgcodecs/doc/reading_and_writing_images.html#imread). Datasets will always be loaded by reading all files within a given directory, optionally filtered by a given suffix string (typically a file-type extension).

Class label annotations are expected to be given in form of JSON files containing a single dictionary which maps the image file name's tail (so without any directory names) to an integer indicating the instance's class.

## Traffic Sign Generation

Traffic sign generation is based on the [German Traffic Sign Recognition Benchmark](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset), the prototype images are generated from [the Wiki Commons Road Signs of Germany](https://commons.wikimedia.org/wiki/Road_signs_of_Germany). A step-by-step introduction on how to train a traffic sign generator with the components in this repository will be available soon!
