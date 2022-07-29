import keras
import tensorflow as tf
import pathlib

DATA_DIR = pathlib.Path(r'D:\SGH\Praca_licencjacka\Kod\Data')
BATCH_SIZE = 32
image_count = len(list(DATA_DIR.glob('*/*.jpg')))
IMG_SIZE = (224, 224)

def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)
    return final_image, label

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=DATA_DIR,
    labels="inferred",  # (labels are generated from the directory structure)
    label_mode='int',
    class_names=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=123,
    validation_split=0.3,
    subset='training',
    interpolation="bilinear",
    follow_links=False,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    directory=DATA_DIR,
    labels="inferred",  # (labels are generated from the directory structure)
    label_mode='int',
    class_names=['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash'],
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=123,
    validation_split=0.3,
    subset='validation',
    interpolation="bilinear",
    follow_links=False,
)

class_names = train_ds.class_names
nr_classes = len(class_names)

val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 5)
val_ds = val_ds.skip(val_batches // 5)

base_model = keras.applications.xception.Xception(include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(nr_classes, activation='softmax')(avg)
model = keras.Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    layer.trainable = False

optimizer = keras.optimizers.SGD(lr=0.02, momentum=0.9, decay=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,  metrics=['accuracy'])
history = model.fit(train_ds, epochs=5, validation_data=val_ds)

for layer in base_model.layers:
    layer.trainable = True

optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, decay=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer,  metrics=['accuracy'])
history = model.fit(train_ds, epochs=5, validation_data=val_ds)