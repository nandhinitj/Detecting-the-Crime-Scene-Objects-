from keras.src.layers import (
    Input, Conv2D, BatchNormalization, Activation,
    UpSampling2D, concatenate, Add
)
from keras.src.models import Model
import numpy as np
from Evaluation import Detect_evaluation


# ================== BASIC BLOCKS ==================

def Conv(x, filters, k=1, s=1, act='swish'):
    x = Conv2D(filters, k, s, padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation(act)(x)
    return x


def Bottleneck(x, filters, shortcut=True):
    y = Conv(x, filters, 1)
    y = Conv(y, filters, 3)
    if shortcut:
        y = Add()([x, y])
    return y


def CSP(x, filters, n):
    x1 = Conv(x, filters, 1)
    x2 = Conv(x, filters, 1)
    for _ in range(n):
        x1 = Bottleneck(x1, filters)
    x = concatenate([x1, x2])
    x = Conv(x, filters, 1)
    return x


# ================== BACKBONE ==================

def CSPDarknet(inputs):
    x = Conv(inputs, 32, 3)
    x = Conv(x, 64, 3, 2)
    x = CSP(x, 64, 1)

    x = Conv(x, 128, 3, 2)
    x = CSP(x, 128, 3)
    P3 = x

    x = Conv(x, 256, 3, 2)
    x = CSP(x, 256, 6)
    P4 = x

    x = Conv(x, 512, 3, 2)
    x = CSP(x, 512, 9)
    P5 = x

    return P3, P4, P5


# ================== NECK (PANet) ==================

def PANet(P3, P4, P5):
    P5_up = UpSampling2D(2)(P5)
    P4 = concatenate([P5_up, P4])
    P4 = CSP(P4, 256, 3)

    P4_up = UpSampling2D(2)(P4)
    P3 = concatenate([P4_up, P3])
    P3 = CSP(P3, 128, 3)

    P3_down = Conv(P3, 256, 3, 2)
    P4 = concatenate([P3_down, P4])
    P4 = CSP(P4, 256, 3)

    P4_down = Conv(P4, 512, 3, 2)
    P5 = concatenate([P4_down, P5])
    P5 = CSP(P5, 512, 3)

    return P3, P4, P5


# ================== DETECTION HEAD ==================

def Detect(x, num_classes, anchors=3):
    return Conv2D(anchors * (num_classes + 5), 1, padding='same')(x)


# YOLOv5 MODEL

def yolov5_model(input_shape=(256, 256, 3), num_classes=3, Act='relu'):
    inputs = Input(input_shape)

    P3, P4, P5 = CSPDarknet(inputs)
    P3, P4, P5 = PANet(P3, P4, P5)

    out_s = Detect(P3, num_classes)
    out_m = Detect(P4, num_classes)
    out_l = Detect(P5, num_classes)

    return Model(inputs, [out_s, out_m, out_l])


def Model_Yolov5(Images, EP=5, BS=2, Act='relu'):
    IMG_SIZE = 256
    CLASSES = 3

    model = yolov5_model((IMG_SIZE, IMG_SIZE, 3), CLASSES, Act)
    model.compile(optimizer='adam', loss=['mse', 'mse', 'mse'])
    model.summary()

    # Resize images
    X = np.zeros((Images.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Images.shape[0]):
        X[i] = np.resize(Images[i], (IMG_SIZE, IMG_SIZE, 3))
    GT = np.zeros((Images.shape[0], IMG_SIZE, IMG_SIZE, 3))
    for i in range(Images.shape[0]):
        GT[i] = np.resize(Images[i], (IMG_SIZE, IMG_SIZE, 3))

    # Dummy ground truth tensors
    y1 = np.zeros((GT.shape[0], IMG_SIZE // 8, IMG_SIZE // 8, 3 * (CLASSES + 5)))
    y2 = np.zeros((GT.shape[0], IMG_SIZE // 16, IMG_SIZE // 16, 3 * (CLASSES + 5)))
    y3 = np.zeros((GT.shape[0], IMG_SIZE // 32, IMG_SIZE // 32, 3 * (CLASSES + 5)))
    model.fit(X, [y1, y2, y3], epochs=EP, batch_size=BS)
    preds = model.predict(X)
    Eval = [Detect_evaluation(y1[n].astype('uint8'), preds[n].astype('uint8')) for n in range(preds.shape[0])]
    EVAl = np.mean(Eval, axis=0)
    return preds, EVAl
