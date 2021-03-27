
### 常见backbone

#### 参数量大{参数量!=模型内存大小（单位为MB）},这里用内存大小来描述

##### VGG16/19


```python
paper: Very Deep Convolutional Networks for Large-Scale Image Recognition
link: https://arxiv.org/pdf/1409.1556.pdf

pre-trained on the ILSVRC-2012-CLS image classification dataset model weights:
link from(https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz

default input size for this model is 224x224，input_shape(224,224,3)
model size(MB): VGG16--528MB  VGG19--549MB
```

##### ResNet152/ResNet152V2, ResNet101/ResNet101V2, ResNet50/ResNet50V2


```python
paper: Deep Residual Learning for Image Recognition
link: https://arxiv.org/pdf/1512.03385.pdf

pre-trained on the ILSVRC-2012-CLS image classification dataset model weights:
link from(https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
http://download.tensorflow.org/models/resnet_v1_152_2016_08_28.tar.gz
http://download.tensorflow.org/models/resnet_v2_152_2017_04_14.tar.gz
http://download.tensorflow.org/models/resnet_v1_101_2016_08_28.tar.gz
http://download.tensorflow.org/models/resnet_v2_101_2017_04_14.tar.gz
http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz

default input size for this model is 224x224，input_shape(224,224,3)
model size(MB): ResNet152--232MB  ResNet152V2--232MB
model size(MB): ResNet101--171MB  ResNet101V2--171MB
model size(MB): ResNet50--98MB  ResNet50V2--98MB
```

##### InceptionResNetV2


```python
pre-trained on the ILSVRC-2012-CLS image classification dataset model weights:
link from(https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz

model size(MB): InceptionResNetV2--215MB
```

##### InceptionV1-V4


```python
paper: Rethinking the Inception Architecture for Computer Vision
link: https://arxiv.org/pdf/1512.00567.pdf

pre-trained on the ILSVRC-2012-CLS image classification dataset model weights:
link from(https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models)
http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz
http://download.tensorflow.org/models/inception_v2_2016_08_28.tar.gz
http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz

default input shape (299, 299, 3)
model size(MB): InceptionV3--92MB
```

##### Xception


```python
paper: Xception: Deep Learning with Depthwise Separable Convolutions
link: https://arxiv.org/pdf/1610.02357.pdf

Xception model with ImageNet weights
https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5
https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5

https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels.h5
https://storage.googleapis.com/tensorflow/keras-applications/xception/xception_weights_tf_dim_ordering_tf_kernels_notop.h5

default input shape (299, 299, 3)
model size(MB): Xception--88MB
```

#### 参数量变化随block块堆积

##### DenseNet121, DenseNet169, DenseNet201


```python
paper: Densely Connected Convolutional Networks
link: https://arxiv.org/pdf/1608.06993.pdf

https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels.h5
https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5
https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet169_weights_tf_dim_ordering_tf_kernels.h5
https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5
https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet201_weights_tf_dim_ordering_tf_kernels.h5
https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5

https://storage.googleapis.com/tensorflow/keras-applications/densenet/
densenet121_weights_tf_dim_ordering_tf_kernels.h5
densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5
densenet169_weights_tf_dim_ordering_tf_kernels.h5
densenet169_weights_tf_dim_ordering_tf_kernels_notop.h5
densenet201_weights_tf_dim_ordering_tf_kernels.h5
densenet201_weights_tf_dim_ordering_tf_kernels_notop.h5

default input shape (224, 224, 3)
model size(MB): DenseNet121--33MB
model size(MB): DenseNet169--57MB
model size(MB): DenseNet201--80MB
```

##### NASNetMobile, NASNetLarge


```python
paper: Learning Transferable Architectures for Scalable Image Recognition
link: https://arxiv.org/pdf/1707.07012.pdf

https://github.com/fchollet/deep-learning-models/releases/download/v0.8/NASNet-mobile-no-top.h5
https://github.com/fchollet/deep-learning-models/releases/download/v0.8/NASNet-mobile.h5
https://github.com/fchollet/deep-learning-models/releases/download/v0.8/NASNet-large-no-top.h5
https://github.com/fchollet/deep-learning-models/releases/download/v0.8/NASNet-large.h5

https://storage.googleapis.com/tensorflow/keras-applications/nasnet/
NASNet-mobile.h5
NASNet-mobile-no-top.h5
NASNet-large.h5
NASNet-large-no-top.h5

the input shapeis by default (331, 331, 3) for NASNetLarge and(224, 224, 3) for NASNetMobile
model size(MB): NASNet-mobile--23MB
model size(MB): NASNet-large--343MB
```

##### MobileNet, MobileNetV2, MobileNetV3 


```python
paper: MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
link: https://arxiv.org/pdf/1704.04861.pdf
paper: MobileNetV2: Inverted Residuals and Linear Bottlenecks
link: https://arxiv.org/pdf/1801.04381.pdf
paper: Searching for MobileNetV3
link: https://arxiv.org/pdf/1905.02244.pdf

alpha数值不一样：0.25，0.5，1.0，1.4
http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz
http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.5_160.tgz
http://download.tensorflow.org/models/mobilenet_v1_2018_02_22/mobilenet_v1_0.25_128.tgz
https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.4_224.tgz
https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz

XML file
large_224_0.75_float，large_224_1.0_float，large_minimalistic_224_1.0_float，
small_224_0.75_float，small_224_1.0_float，small_minimalistic_224_1.0_float
https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v3/
765b44a33ad4005b3ac83185abf1d0ebe7b4d1071996dd51a2c2ca2424570e20
59e551e166be033d707958cf9e29a6a7037116398e07f018c0005ffcb0406831
675e7b876c45c57e9e63e6d90a36599ca2c33aed672524d1d0b4431808177695
cb65d4e5be93758266aa0a7f2c6708b74d2fe46f1c1f38057392514b0df1d673
8768d4c2e7dee89b9d02b2d03d65d862be7100780f875c06bcab93d76641aa26
99cd97fb2fcdad2bf028eb838de69e3720d4e357df3f7a6361f3a288857b1051

pre-trained model download
https://github.com/qubvel/efficientnet/releases

model size(MB): mobilenet--16MB
model size(MB): mobilenetV2--14MB
model size(MB): mobilenet_v3_large_1.0_224--217MB
model size(MB): mobilenet_v3_large_0.75_224--155MB
model size(MB): mobilenet_v3_large_minimalistic_1.0_224--209MB
model size(MB): mobilenet_v3_small_1.0_224--66MB
model size(MB): mobilenet_v3_small_0.75_224--44MB
model size(MB): mobilenet_v3_small_minimalistic_1.0_224--65MB
```

##### EfficientNetB0-B7


```python
paper: EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks
link: https://arxiv.org/pdf/1905.11946.pdf

XML file
b0-b7
https://storage.googleapis.com/keras-applications/efficientnet/
902e53a9f72be733fc0bcb005b3ebbac50bc09e76180e00e4465e1a485ddc09d
1d254153d4ab51201f1646940f01854074c4e6b3e1f6a1eea24c589628592432
b15cce36ff4dcbd00b6dd88e7857a6ad111f8e2ac8aa800a7a99e3239f7bfb39
ffd1fdc53d0ce67064dc6a9c7960ede0af6d107764bb5b1abb91932881670226
18c95ad55216b8f92d7e70b3a046e2fcebc24e6d6c33eaebbd558eafbeedf1ba
ace28f2a6363774853a83a0b21b9421a38879255a25d3c92d5e44e04ae6cec6f
165f6e37dce68623721b423839de8be59ecce42647a20130c1f39a5d4cb75743
8c03f828fec3ef71311cd463b6759d99cbcfe4450ddf6f3ad90b1b398090fe4a

参数量（millions）: 
B0--5.3M
B1--7.8M
B2--9.2M
B3--12M  (约48MB)
B4--19M
B5--30M
B6--43M
B7--66M
```
