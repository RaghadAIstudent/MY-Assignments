import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

class_names = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
MODEL_PATH = "intel_model.h5"

if not os.path.exists(MODEL_PATH):
    print(f"خطأ: ملف النموذج {MODEL_PATH} غير موجود!")
    exit()

print("جاري تحميل النموذج المدرب...")
model = tf.keras.models.load_model(MODEL_PATH)
print("تم تحميل النموذج بنجاح! يمكنك الآن تجربة الصور.\n")

while True:
    img_path = input("اسحب الصورة إلى هنا أو الصق مسارها (اكتب exit للخروج): ").strip().strip('"').strip("'")
    
    if img_path.lower() in ['exit', 'خروج', 'q']:
        print("تم إنهاء البرنامج.")
        break
        
    if not os.path.exists(img_path):
        print("المسار غير صحيح أو الصورة غير موجودة، حاول مجدداً.\n")
        continue

    try:
        # قراءة وتجهيز الصورة
        img = tf.keras.utils.load_img(img_path, target_size=(150, 150))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        # التنبؤ
        preds = model.predict(img_array, verbose=0)[0]
        predicted_class = class_names[np.argmax(preds)]
        confidence = 100 * np.max(preds)

        print("-" * 40)
        print(f"النتيجة: {predicted_class}")
        print(f"نسبة الثقة: {confidence:.2f}%")
        print("-" * 40)

        # عرض الصورة المجرّبة مع التوقع في نافذة
        plt.imshow(img)
        plt.title(f"{predicted_class} ({confidence:.1f}%)")
        plt.axis('off')
        plt.show()

    except Exception as e:
        print(f"حدث خطأ أثناء قراءة الصورة: {e}\n")