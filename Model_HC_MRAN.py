import tensorflow as tf
from tensorflow.keras import layers, Model, Input
import numpy as np
from Evaluation import evaluation


# Hybrid Convolution Block
def hybrid_conv_block(x, filters):
    conv1 = layers.Conv2D(filters, 1, padding="same", activation="relu")(x)
    conv3 = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
    conv5 = layers.Conv2D(filters, 5, padding="same", activation="relu")(x)

    x = layers.Concatenate()([conv1, conv3, conv5])
    x = layers.BatchNormalization()(x)
    return x


# Residual Block (unchanged)
def residual_block(x, filters, stride=1):
    shortcut = x

    x = layers.Conv2D(filters, 3, strides=stride, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)

    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


# Channel Attention (SE Block)
def channel_attention(x, ratio=8):
    filters = x.shape[-1]

    se = layers.GlobalAveragePooling2D()(x)
    se = layers.Dense(filters // ratio, activation="relu")(se)
    se = layers.Dense(filters, activation="sigmoid")(se)
    se = layers.Reshape((1, 1, filters))(se)

    return layers.Multiply()([x, se])


# Spatial Attention
def spatial_attention(x):
    avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
    max_pool = tf.reduce_max(x, axis=-1, keepdims=True)

    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attention = layers.Conv2D(1, 7, padding="same", activation="sigmoid")(concat)

    return layers.Multiply()([x, attention])


# Hybrid Attention Module
def hybrid_attention_module(x):
    x = channel_attention(x)
    x = spatial_attention(x)
    return x


# Multiscale Block
def multiscale_block(x, filters):
    # Original scale
    x1 = hybrid_conv_block(x, filters)

    # Downsample
    x2 = layers.MaxPooling2D()(x)
    x2 = hybrid_conv_block(x2, filters)
    x2 = layers.UpSampling2D()(x2)

    # Fuse
    x = layers.Concatenate()([x1, x2])
    x = layers.Conv2D(filters, 1, padding="same")(x)

    return x


# HC-MRAN MODEL
def HC_MRAN(input_shape, num_classes, HN):
    inputs = Input(shape=input_shape)

    # Initial Layer
    x = layers.Conv2D(64, 7, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    # Stage 1
    x = multiscale_block(x, 64)
    x = residual_block(x, 64)
    x = hybrid_attention_module(x)

    # Stage 2
    x = multiscale_block(x, 128)
    x = residual_block(x, 128, stride=2)
    x = hybrid_attention_module(x)

    # Stage 3
    x = multiscale_block(x, 256)
    x = residual_block(x, 256, stride=2)
    x = hybrid_attention_module(x)

    # Stage 4
    x = multiscale_block(x, 512)
    x = residual_block(x, 512, stride=2)
    x = hybrid_attention_module(x)

    # Classification Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(HN, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return Model(inputs, outputs)


def Model_HC_MRAN(train_data, train_target, test_data, test_target, BS=None, HN=None):
    if BS is None:
        BS = 4
    if HN is None:
        HN = 512

    input_shape = (224, 224, 3)
    num_classes = train_target.shape[-1]

    X_train = train_data.reshape((-1, 224, 224, 3))
    X_test = test_data.reshape((-1, 224, 224, 3))

    model = HC_MRAN(input_shape, num_classes, HN)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=['accuracy']
    )

    model.summary()

    model.fit(
        X_train, train_target,
        batch_size=BS,
        epochs=10,
        validation_data=(X_test, test_target),
        verbose=1
    )

    pred = model.predict(X_test)

    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = pred.astype('int')
    Eval = evaluation(test_target, pred)
    return Eval, pred
