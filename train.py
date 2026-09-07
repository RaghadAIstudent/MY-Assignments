import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ==========================================
# 1. فحص الإصدار وتحديد المسارات
# ==========================================
print("TensorFlow Version:", tf.__version__)

# مسار مجلد البيانات النسبي الموجود بجانب ملف الكود
base_dir = "data"
train_dir = os.path.join(base_dir, "seg_train", "seg_train")
test_dir = os.path.join(base_dir, "seg_test", "seg_test")
pred_dir = os.path.join(base_dir, "seg_pred", "seg_pred")

# في حال لم يكن المجلد مكرراً بعد فك الضغط
if not os.path.exists(train_dir):
    train_dir = os.path.join(base_dir, "seg_train")
    test_dir = os.path.join(base_dir, "seg_test")
    pred_dir = os.path.join(base_dir, "seg_pred")

print(f"مسار التدريب: {train_dir}")
print(f"مسار الاختبار: {test_dir}")

# ==========================================
# 2. تحميل وتجهيز البيانات (Data Pipeline)
# ==========================================
IMG_SIZE = (150, 150)
BATCH_SIZE = 32

print("\n--- جاري تحميل مجموعات البيانات ---")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=True
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

class_names = train_dataset.class_names
num_classes = len(class_names)
print(f"\nعدد الفئات: {num_classes}")
print("أسماء الفئات:", class_names)

# تحسين سرعة وسلاسة القراءة من القرص الصلب
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# ==========================================
# 3. عرض عينات من الصور وحفظها
# ==========================================
plt.figure(figsize=(10, 6))
for i, category in enumerate(class_names):
    cat_dir = os.path.join(train_dir, category)
    if os.path.exists(cat_dir):
        images = os.listdir(cat_dir)
        if images:
            img = Image.open(os.path.join(cat_dir, images[0]))
            plt.subplot(2, 3, i + 1)
            plt.imshow(img)
            plt.title(category)
            plt.axis('off')
plt.tight_layout()
plt.savefig("sample_classes.png")
plt.close()
print("تم حفظ نموذج من صور الأصناف في ملف: sample_classes.png")

# ==========================================
# 4. بناء هيكل النموذج (CNN Architecture)
# ==========================================
model = keras.Sequential([
    # توحيد قيم البيكسل بين 0 و 1
    layers.Rescaling(1./255, input_shape=(150, 150, 3)),
    
    # زيادة البيانات لمنع الـ Overfitting
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    
    # بلوك الالتفاف الأول
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # بلوك الالتفاف الثاني
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # بلوك الالتفاف الثالث
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    # بلوك الالتفاف الرابع
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    
    # الطبقات الكثيفة (Dense Layers)
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 5. تدريب النموذج (Model Training)
# ==========================================
EPOCHS = 12
print("\n--- بدء التدريب ---")
history = model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=EPOCHS
)

# ==========================================
# 6. رسم وحفظ منحنيات الدقة والخسارة
# ==========================================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
plt.close()
print("\nتم حفظ منحنيات التدريب في ملف: training_curves.png")

# ==========================================
# 7. التقييم وحساب مصفوفة الالتباس
# ==========================================
print("\n--- تقييم النموذج على بيانات الاختبار ---")
loss, accuracy = model.evaluate(test_dataset)
print(f"دقة النموذج النهائية: {accuracy * 100:.2f}%")

# جمع التوقعات والقيم الفعلية
y_true = []
y_pred = []
for images, labels in test_dataset:
    preds = model.predict(images, verbose=0)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

print("\n--- تقرير التصنيف (Classification Report) ---")
print(classification_report(y_true, y_pred, target_names=class_names))

# مصفوفة الالتباس
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('التوقع (Predicted)')
plt.ylabel('الحقيقي (Actual)')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
print("تم حفظ مصفوفة الالتباس في ملف: confusion_matrix.png")

# ==========================================
# 8. حفظ النموذج وتجربة التنبؤ على صورة جديدة
# ==========================================
model.save("intel_model.h5")
print("تم حفظ النموذج المدرب بنجاح باسم: intel_model.h5")

# تجربة صورة عشوائية من مجلد seg_pred إذا كان متوفراً
if os.path.exists(pred_dir):
    pred_images = [f for f in os.listdir(pred_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if pred_images:
        test_img_path = os.path.join(pred_dir, pred_images[0])
        img = tf.keras.utils.load_img(test_img_path, target_size=IMG_SIZE)
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # جعلها دفعة بحجم 1

        prediction = model.predict(img_array, verbose=0)
        predicted_class = class_names[np.argmax(prediction[0])]
        confidence = 100 * np.max(prediction[0])

        print(f"\nتجربة تنبؤ على الصورة ({pred_images[0]}):")
        print(f"الصنف المتوقع: {predicted_class} (بنسبة ثقة: {confidence:.2f}%)")