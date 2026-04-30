import numpy as np
import cv2 as cv
from keras.src.applications.mobilenet import MobileNet
from keras.src.models import Model
from keras.src.layers import GlobalAveragePooling2D, Dense, Input
from keras.src.optimizers import Adam
from Evaluation import evaluation


# Mobilenet
def Model_Mobilenet(train_data, train_tar, test_data, test_tar, HN=None, BS=None):
    if HN is None :
        HN =50
    if BS is None:
        BS = 64
    IMG_SIZE = (224, 224)
    num_classes = train_tar.shape[-1]

    # Resize input images properly
    def preprocess_images(data):
        processed = np.zeros((data.shape[0], IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.float32)
        for i in range(data.shape[0]):
            resized = cv.resize(data[i], IMG_SIZE)  # cv.resize expects (width, height)
            if resized.ndim == 2:  # if grayscale
                resized = cv.cvtColor(resized, cv.COLOR_GRAY2RGB)
            processed[i] = resized / 255.0  # normalize
        return processed

    Train_x = preprocess_images(train_data)
    Test_x = preprocess_images(test_data)

    # Build model
    base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(HN, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    model.fit(Train_x, train_tar, epochs=20, batch_size=BS, steps_per_epoch=10,
              validation_data=(Test_x, test_tar))
    pred = model.predict(Test_x)
    avg = np.mean(pred)
    pred[pred >= avg] = 1
    pred[pred < avg] = 0
    pred = pred.astype('int')
    Eval = evaluation(test_tar, pred)
    return Eval, pred