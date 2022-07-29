import os
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import resnet
from tensorflow.keras.applications.resnet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten, Input, AveragePooling2D
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


SEED = 1
BATCH_SIZE = 32
DATA_DIR = Path('output_double')
ALL_DATA_DIR = Path(r'Data')
INIT_LR = 1e-4
NUM_EPOCHS = 5

MODEL_PATH = "resnet3.model"

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

base_model = resnet.ResNet152(weights='imagenet', include_top = False, input_shape = (224,224,3))

headModel = base_model.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(256, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(n_classes, activation="softmax")(headModel)

model = Model(inputs=base_model.input, outputs=headModel)

for layer in base_model.layers:
  layer.trainable = False

opt = Adam(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training model...")
H = model.fit_generator(
	traingen,
	steps_per_epoch=traingen.samples // BATCH_SIZE,
	validation_data=validgen,
	validation_steps=validgen.samples // BATCH_SIZE,
	epochs=NUM_EPOCHS)



print("[INFO] evaluating network...")


true_classes = testgen.classes
class_indices = traingen.class_indices
class_indices = dict((v,k) for k,v in class_indices.items())


testgen.reset()
predIdxs = model.predict(testgen)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testgen.classes, predIdxs,
	target_names=testgen.class_indices.keys(), digits=4))


resnet_acc = accuracy_score(true_classes, predIdxs)
print("VGG16 Model Accuracy without Fine-Tuning: {:.2f}%".format(resnet_acc * 100))

print("[INFO] saving model...")
model.save(MODEL_PATH, save_format="h5")
#
#
# n_steps = traingen.samples // BATCH_SIZE
# n_val_steps = validgen.samples // BATCH_SIZE
# n_epochs = 5
#
# x = Flatten()(base_model.output)
# x = Dense(1000, activation='relu')(x)
# predictions = Dense(6, activation = 'softmax')(x)
#
# head_model = Model(inputs = base_model.input, outputs = predictions)
# head_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# history = head_model.fit(traingen,
#                             batch_size=BATCH_SIZE,
#                             epochs=n_epochs,
#                             validation_data=validgen,
#                             steps_per_epoch=n_steps,
#                             validation_steps=n_val_steps,
#                             verbose=1)

for layer in base_model.layers:
  layer.trainable = True

opt = Adam(lr=INIT_LR, decay=INIT_LR / NUM_EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] training model...")
H = model.fit_generator(
	traingen,
	steps_per_epoch=traingen.samples // BATCH_SIZE,
	validation_data=validgen,
	validation_steps=validgen.samples // BATCH_SIZE,
	epochs=NUM_EPOCHS)



print("[INFO] evaluating network...")


true_classes = testgen.classes
class_indices = traingen.class_indices
class_indices = dict((v,k) for k,v in class_indices.items())


testgen.reset()
predIdxs = model.predict(testgen)

predIdxs = np.argmax(predIdxs, axis=1)

print(classification_report(testgen.classes, predIdxs,
	target_names=testgen.class_indices.keys(), digits=4))


resnet_acc = accuracy_score(true_classes, predIdxs)
print("VGG16 Model Accuracy without Fine-Tuning: {:.2f}%".format(resnet_acc * 100))

print("[INFO] saving model...")
model.save(MODEL_PATH, save_format="h5")
