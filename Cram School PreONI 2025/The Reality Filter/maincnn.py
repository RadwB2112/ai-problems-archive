import os
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array


TRAIN_CSV = 'train_data.csv'
TRAIN_DIR = 'train'
TEST_DIR  = 'test'
IMG_SIZE   = (128, 128)
BATCH_SIZE = 32


# Subtask 1
image = Image.open(f"{TRAIN_DIR}/129.jpg")
width, height = image.size
subtask1 = width * height

# subtask2
df = pd.read_csv('train_data.csv')
counter = df['Label'].value_counts()
min, max = counter.min(), counter.max()
subtask2 = min / max


print(f"Subtask1 (pixels): {subtask1}")
print(f"Subtask2 (ratio):  {subtask2:.4f}")

### === Subtask 3: CNN Training ===

df['filename'] = df['CodeID'].astype(str) + '.jpg'
df['label_str'] = df['Label'].astype(str)  

# Data augmentation + split
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

train_gen = datagen.flow_from_dataframe(
    df, TRAIN_DIR,
    x_col='filename', y_col='label_str',
    subset='training',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary'
)

val_gen = datagen.flow_from_dataframe(
    df, TRAIN_DIR,
    x_col='filename', y_col='label_str',
    subset='validation',
    target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='binary'
)

# Build small CNN

model = models.Sequential([
    layers.Input((*IMG_SIZE, 3)),
    
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),
    
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.3),

    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(),
    layers.Dropout(0.4),

    layers.Flatten(),
    
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Dense(1, activation='sigmoid')
])

model.compile(
  optimizer = tf.keras.optimizers.RMSprop(learning_rate=1e-4),
  loss='binary_crossentropy',
  metrics=['accuracy',
           tf.keras.metrics.Precision(),
           tf.keras.metrics.Recall(),
           tf.keras.metrics.AUC()]
)
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_auc', mode='max', patience=3, restore_best_weights=True)

from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(df['Label']),
    y=df['Label']
)

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=30,
    callbacks=[es],
    class_weight=dict(enumerate(class_weights))
)


# Subtask 3
test_files = sorted(
    [f for f in os.listdir(TEST_DIR) if f.endswith('.jpg')],
    key=lambda x: int(x.split('.')[0])
)
test_ids = [os.path.splitext(f)[0] for f in test_files]

def load_and_preprocess_test_image(path):
    img = load_img(path, target_size=IMG_SIZE)
    img = img_to_array(img) / 255.0
    return img

X_test = np.array([load_and_preprocess_test_image(os.path.join(TEST_DIR, f)) for f in test_files])
preds = model.predict(X_test)
pred_labels = (preds > 0.5).astype(int).flatten()


# output for subtask 1 + 2
rows = []
rows.append({'subtaskID': 1, 'datapointID': 0, 'answer': int(subtask1)})
rows.append({'subtaskID': 2, 'datapointID': 0, 'answer': subtask2})

# output for subtask 3
for code_id, label in zip(test_ids, pred_labels):
    rows.append({'subtaskID': 3, 'datapointID': int(code_id), 'answer': int(label)})

# save 
out_df = pd.DataFrame(rows, columns=['subtaskID', 'datapointID', 'answer'])
out_df.to_csv('output.csv', index=False)

