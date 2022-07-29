import os
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import xception
from tensorflow.keras.applications.xception import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, AveragePooling2D, GlobalAveragePooling2D
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


SEED = 1
BATCH_SIZE = 16
DATA_DIR = Path('output_double')
TRAIN_DATA_DIR = Path('Augmented_preview')
ALL_DATA_DIR = Path(r'Data')
INIT_LR = 1e-4
NUM_EPOCHS = 5
based_model_last_block_layer_number = 126

MODEL_PATH = "xceptionnew2.model"

train_generator = ImageDataGenerator(rotation_range=40,
                                     brightness_range=[0.1, 0.7],
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     validation_split=0.15,
                                     preprocessing_function=preprocess_input)

test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

classes = sorted(os.listdir(ALL_DATA_DIR))
n_classes = len(classes)

os.makedirs(os.path.join(os.path.abspath(TRAIN_DATA_DIR), '../preview'), exist_ok=True)
traingen = train_generator.flow_from_directory(DATA_DIR/'train',
                                               target_size=(299, 299),
                                               class_mode='categorical',
                                               classes=classes,
                                               subset='training',
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               seed=SEED)

validgen = train_generator.flow_from_directory(DATA_DIR/'train', target_size=(299, 299),
                                               class_mode='categorical',
                                               classes=classes,
                                               subset='validation',
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               seed=SEED)

testgen = test_generator.flow_from_directory(DATA_DIR/'test',
                                             target_size=(299, 299),
                                             class_mode=None,
                                             classes=classes,
                                             batch_size=1,
                                             shuffle=False,
                                             seed=123)

base_model = xception.Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(n_classes, activation='softmax')(x)

model = Model(base_model.input, predictions)
print(model.summary())

for layer in base_model.layers:
        layer.trainable = False

model.compile(optimizer='nadam',loss='categorical_crossentropy', metrics=['accuracy'])

top_weights_path = os.path.join(os.path.abspath(MODEL_PATH), 'top_model_weights_xceptionnew2.h5')
callbacks_list = [ModelCheckpoint(top_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
                  EarlyStopping(monitor='val_acc', patience=5, verbose=0)]

model.fit_generator(train_generator,
                        samples_per_epoch=train_generator.nb_sample,
                        nb_epoch=NUM_EPOCHS / 5,
                        validation_data=validgen,
                        nb_val_samples=validgen.nb_sample,
                        callbacks=callbacks_list)

print("\nStarting to Fine Tune Model\n")

for layer in model.layers[:based_model_last_block_layer_number]:
    layer.trainable = False
for layer in model.layers[based_model_last_block_layer_number:]:
    layer.trainable = True

    # add the best weights from the train top model
    # at this point we have the pre-train weights of the base model and the trained weight of the new/added top model
    # we re-load model weights to ensure the best epoch is selected and not the last one.
model.load_weights(top_weights_path)

model.compile(optimizer='nadam',loss='categorical_crossentropy', metrics=['accuracy'])


# based_model_last_block_layer_number points to the layer in your model you want to train.
# For example if you want to train the last block of a 19 layer VGG16 model this should be 15
# If you want to train the last Two blocks of an Inception model it should be 172
# layers before this number will used the pre-trained weights, layers above and including this number
# will be re-trained based on the new data.
for layer in model.layers[:based_model_last_block_layer_number]:
    layer.trainable = False
for layer in model.layers[based_model_last_block_layer_number:]:
    layer.trainable = True

model.fit_generator(train_generator,
                        samples_per_epoch=train_generator.nb_sample,
                        nb_epoch=NUM_EPOCHS,
                        validation_data=validgen,
                        nb_val_samples=validgen.nb_sample,
                        callbacks=callbacks_list)

final_weights_path = os.path.join(os.path.abspath(MODEL_PATH), 'model_weights_xceptionnew2.h5')
callbacks_list = [
    ModelCheckpoint(final_weights_path, monitor='val_acc', verbose=1, save_best_only=True),
    EarlyStopping(monitor='val_loss', patience=5, verbose=0)
]

model.fit_generator(train_generator,
                        samples_per_epoch=train_generator.nb_sample,
                        nb_epoch=NUM_EPOCHS / 5,
                        validation_data=validgen,
                        nb_val_samples=validgen.nb_sample,
                        callbacks=callbacks_list)

model_json = model.to_json()
with open(os.path.join(os.path.abspath(MODEL_PATH), 'model.json'), 'w') as json_file:
    json_file.write(model_json)
