import os
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score


SEED = 123
BATCH_SIZE = 32
DATA_DIR = Path('output_double')
ALL_DATA_DIR = Path(r'Data')


train_generator = ImageDataGenerator(rotation_range=40,
                                     brightness_range=[0.1, 0.7],
                                     width_shift_range=0.2,
                                     height_shift_range=0.2,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     validation_split=0.15,
                                     preprocessing_function=preprocess_input)  # VGG16 preprocessing

test_generator = ImageDataGenerator(preprocessing_function=preprocess_input)  # VGG16 preprocessing

classes = sorted(os.listdir(ALL_DATA_DIR))

traingen = train_generator.flow_from_directory(DATA_DIR/'train',
                                               target_size=(224, 224),
                                               class_mode='categorical',
                                               classes=classes,
                                               subset='training',
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               seed=SEED)

validgen = train_generator.flow_from_directory(DATA_DIR/'train',
                                               target_size=(224, 224),
                                               class_mode='categorical',
                                               classes=classes,
                                               subset='validation',
                                               batch_size=BATCH_SIZE,
                                               shuffle=True,
                                               seed=SEED)

testgen = test_generator.flow_from_directory(DATA_DIR/'test',
                                             target_size=(224, 224),
                                             class_mode=None,
                                             classes=classes,
                                             batch_size=1,
                                             shuffle=False,
                                             seed=123)

# pretrained_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

# for layer in pretrained_model.layers[:15]:
#     layer.trainable = False
#
# for layer in pretrained_model.layers[15:]:
#     layer.trainable = True
#
# last_layer = pretrained_model.get_layer('block5_pool')
# last_output = last_layer.output
#
# x = GlobalMaxPooling2D()(last_output)
# x = Dense(512, activation='relu')(x)
# x = Dropout(0.5)(x)
# x = Dense(1, activation='sigmoid')(x)

# model = Sequential()
# model.add(VGG16(include_top = False, weights = weights, input_shape = (224,224,3)))
# model.add(Flatten())
# model.add(Dense(512, activation = 'relu'))
# model.add(Dense(5, activation = 'softmax'))
# model.compile(loss='categorical_crossentropy',
#               optimizer='sgd',
#               metrics=['accuracy'])


def create_model(input_shape, n_classes, optimizer='rmsprop', fine_tune=0):
    """
    Compiles a model integrated with VGG16 pretrained layers

    input_shape: tuple - the shape of input images (width, height, channels)
    n_classes: int - number of classes for the output layer
    optimizer: string - instantiated optimizer to use for training. Defaults to 'RMSProp'
    fine_tune: int - The number of pre-trained layers to unfreeze.
                If set to 0, all pretrained layers will freeze during training
    """

    # Pretrained convolutional layers are loaded using the Imagenet weights.
    # Include_top is set to False, in order to exclude the model's fully-connected layers.
    conv_base = VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=input_shape)

    # Defines how many layers to freeze during training.
    # Layers in the convolutional base are switched from trainable to non-trainable
    # depending on the size of the fine-tuning parameter.
    if fine_tune > 0:
        for layer in conv_base.layers[:-fine_tune]:
            layer.trainable = False
    else:
        for layer in conv_base.layers:
            layer.trainable = False

    # Create a new 'top' of the model (i.e. fully-connected layers).
    # This is 'bootstrapping' a new top_model onto the pretrained layers.
    top_model = conv_base.output
    top_model = Flatten(name="flatten")(top_model)
    top_model = Dense(4096, activation='relu')(top_model)
    top_model = Dense(1072, activation='relu')(top_model)
    top_model = Dropout(0.2)(top_model)
    output_layer = Dense(n_classes, activation='softmax')(top_model)

    # Group the convolutional base and new fully-connected layers into a Model object.
    model = Model(inputs=conv_base.input, outputs=output_layer)

    # Compiles the model for training.
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


input_shape = (224, 224, 3)
optim_1 = Adam(learning_rate=0.001)
n_classes = len(classes)

n_steps = traingen.samples // BATCH_SIZE
n_val_steps = validgen.samples // BATCH_SIZE
n_epochs = 100

# First we'll train the model without Fine-tuning
vgg_model = create_model(input_shape, n_classes, 'Adam', fine_tune=0)

from livelossplot.inputs.keras import PlotLossesCallback

plot_loss_1 = PlotLossesCallback()

# ModelCheckpoint callback - save best weights
tl_checkpoint_1 = ModelCheckpoint(filepath='tl_model_v1.weights.best.hdf5',
                                  save_best_only=True,
                                  verbose=1)

# EarlyStopping
early_stop = EarlyStopping(monitor='val_loss',
                           patience=10,
                           restore_best_weights=True,
                           mode='min')


vgg_history = vgg_model.fit(traingen,
                            batch_size=BATCH_SIZE,
                            epochs=n_epochs,
                            validation_data=validgen,
                            steps_per_epoch=n_steps,
                            validation_steps=n_val_steps,
                            callbacks=[tl_checkpoint_1, early_stop, plot_loss_1],
                            verbose=1)

vgg_model.load_weights('tl_model_v1.weights.best.hdf5') # initialize the best trained weights

true_classes = testgen.classes
class_indices = traingen.class_indices
class_indices = dict((v,k) for k,v in class_indices.items())

vgg_preds = vgg_model.predict(testgen)
vgg_pred_classes = np.argmax(vgg_preds, axis=1)

vgg_acc = accuracy_score(true_classes, vgg_pred_classes)
print("VGG16 Model Accuracy without Fine-Tuning: {:.2f}%".format(vgg_acc * 100))