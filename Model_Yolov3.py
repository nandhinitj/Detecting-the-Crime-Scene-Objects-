import keras
from keras.src.models import Model
from keras.src.layers import (
    Input, Conv2D, BatchNormalization, LeakyReLU,
    ZeroPadding2D, UpSampling2D, add, concatenate
)
import numpy as np
from Evaluation import Detect_evaluation


# Convolutional block: Conv -> BatchNorm -> LeakyReLU
def conv_bn_leaky(x, filters, kernel_size, strides=1):
    if strides == 2:  # Darknet style padding for downsampling
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)
        padding = 'valid'
    else:
        padding = 'same'
    x = Conv2D(filters, kernel_size, strides=strides,
               padding=padding, use_bias=False)(x)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x


# Residual block: 2 conv + skip connection
def residual_block(x, filters):
    skip = x
    x = conv_bn_leaky(x, filters // 2, 1)
    x = conv_bn_leaky(x, filters, 3)
    x = add([skip, x])
    return x


# Darknet-53 backbone
def darknet53(input_tensor):
    x = conv_bn_leaky(input_tensor, 32, 3)
    x = conv_bn_leaky(x, 64, 3, strides=2)

    # 1 residual block
    x = residual_block(x, 64)

    # 2 residual blocks
    x = conv_bn_leaky(x, 128, 3, strides=2)
    for _ in range(2):
        x = residual_block(x, 128)

    # 8 residual blocks
    x = conv_bn_leaky(x, 256, 3, strides=2)
    for _ in range(8):
        x = residual_block(x, 256)
    route1 = x  # scale: 52x52

    # 8 residual blocks
    x = conv_bn_leaky(x, 512, 3, strides=2)
    for _ in range(8):
        x = residual_block(x, 512)
    route2 = x  # scale: 26x26

    # 4 residual blocks
    x = conv_bn_leaky(x, 1024, 3, strides=2)
    for _ in range(4):
        x = residual_block(x, 1024)
    route3 = x  # scale: 13x13

    return route1, route2, route3


# Build detection head blocks for YOLOv3
def yolo_head(x, filters):
    x = conv_bn_leaky(x, filters, 1)
    x = conv_bn_leaky(x, filters * 2, 3)
    x = conv_bn_leaky(x, filters, 1)
    x = conv_bn_leaky(x, filters * 2, 3)
    x = conv_bn_leaky(x, filters, 1)
    return x


def yolo_output(x, filters, num_classes):
    # Final conv: predicts (num_anchors * (num_classes + 5)) outputs
    return Conv2D(filters * (num_classes + 5), 1, padding='same')(x)


def YoloV3(input_shape=(416, 416, 3), num_classes=80):
    inputs = Input(input_shape)

    # Darknet-53 backbone
    route1, route2, route3 = darknet53(inputs)

    # YOLO head for scale 13x13
    x = yolo_head(route3, 512)
    out1 = yolo_output(x, 3, num_classes)

    # Up & merge 26x26
    x = conv_bn_leaky(x, 256, 1)
    x = UpSampling2D(2)(x)
    x = concatenate([x, route2])
    x = yolo_head(x, 256)
    out2 = yolo_output(x, 3, num_classes)

    # Up & merge 52x52
    x = conv_bn_leaky(x, 128, 1)
    x = UpSampling2D(2)(x)
    x = concatenate([x, route1])
    x = yolo_head(x, 128)
    out3 = yolo_output(x, 3, num_classes)

    model = Model(inputs, [out1, out2, out3])
    return model


def Model_Yolov3(Images, HN=None, sol=None):
    if sol is None:
        sol = [4, 50, 0, 5, 0]
    if HN is None:
        HN = 64
    IMG_SIZE = 416
    classes = 3
    optimizer = ['SGD', 'Adam', 'RMSprop', 'Adagrad', 'Adadelta']
    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    # Resize Images
    Train_X = np.zeros((Images.shape[0], *input_shape))
    for i in range(Images.shape[0]):
        Train_X[i] = np.resize(Images[i], input_shape)

    Train_Y = np.zeros((Images.shape[0], *input_shape))
    for i in range(Images.shape[0]):
        Train_Y[i] = np.resize(Images[i], input_shape)

    model = YoloV3(input_shape=input_shape, num_classes=classes)
    model.compile(optimizer=optimizer[int(sol[2])], loss=keras.losses.binary_crossentropy, metrics=['accuracy'])
    model.summary()
    model.fit(Train_X, Train_Y, epochs=sol[1], steps_per_epoch=2, verbose="auto")
    Predict = model.predict(Train_X)
    Eval = [Detect_evaluation(Train_Y[n].astype('uint8'), Predict[n].astype('uint8')) for n in range(Predict.shape[0])]
    EVAl = np.mean(Eval, axis=0)
    return Predict, EVAl
