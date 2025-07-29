import os
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# ==== CONFIG ====
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
CSV_PATH = "HAM10000_metadata.csv"
IMG_DIR_1 = "HAM10000_images_part_1images1"
IMG_DIR_2 = "HAM10000_images_part_2"
MODEL_SAVE_PATH = "skin_cancer_model.h5"

# ==== STEP 1: LOAD CSV & VALIDATE IMAGE PATHS ====
df = pd.read_csv(CSV_PATH)

def get_img_path(img_id):
    fname = img_id + ".jpg"
    for folder in [IMG_DIR_1, IMG_DIR_2]:
        full_path = os.path.join(folder, fname)
        if os.path.exists(full_path):
            return full_path
    return None

df['image_path'] = df['image_id'].apply(get_img_path)
df = df[df['image_path'].notnull()]  # Remove missing images
df['label'] = df['dx']  # Use class names directly

print(f"✅ Valid image rows: {len(df)}")

# ==== STEP 2: TRAIN/VAL SPLIT ====
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# ==== STEP 3: IMAGE DATA GENERATORS ====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col='image_path',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_data = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col='image_path',
    y_col='label',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

num_classes = len(train_data.class_indices)
print(f"🧪 Classes detected: {num_classes} ➤ {train_data.class_indices}")

# ==== STEP 4: MODEL ====
base_model = MobileNetV2(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                         include_top=False,
                         weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ==== STEP 5: TRAIN ====
checkpoint = ModelCheckpoint(MODEL_SAVE_PATH,
                             monitor='val_accuracy',
                             save_best_only=True,
                             mode='max')

model.fit(train_data,
          validation_data=val_data,
          epochs=EPOCHS,
          callbacks=[checkpoint])

print("✅ Training complete. Model saved as:", MODEL_SAVE_PATH)
