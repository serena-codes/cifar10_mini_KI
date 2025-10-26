# 📦 Bibliotheken importieren
import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping
from PIL import Image, ImageFilter
import numpy as np

# 📊 CIFAR-10 Datensatz laden und vorbereiten
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# 🧠 Modell definieren
model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
model.summary()

# ⚙️ Kompilieren und Trainingskonfiguration
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# 🚀 Modell trainieren
model.fit(
    x_train, y_train,
    epochs=50,
    batch_size=64,
    validation_split=0.1,
    callbacks=[early_stop]
)

# 🧪 Modell evaluieren
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Testgenauigkeit: {test_acc:.2f}")

# 🏷️ Klassenbezeichnungen
class_names = ['Flugzeug', 'Auto', 'Vogel', 'Katze', 'Reh', 'Hund', 'Frosch', 'Pferd', 'Schiff', 'LKW']

# 🔍 Einzelne Vorhersagen anzeigen
for i in range(5):
    prediction = model.predict(x_test[i:i+1])
    predicted_label = prediction[0].argmax()
    true_label = y_test[i][0]
    plt.figure(figsize=(2, 2))
    plt.imshow(x_test[i])
    plt.title(f"Vorhersage: {class_names[predicted_label]}\nWahr: {class_names[true_label]}")
    plt.axis('off')
    plt.show()

# 🖼️ Mehrere Vorhersagen nebeneinander anzeigen
fig, axes = plt.subplots(1, 5, figsize=(12, 3))
for i in range(5):
    prediction = model.predict(x_test[i:i+1])
    predicted_label = prediction[0].argmax()
    true_label = y_test[i][0]
    axes[i].imshow(x_test[i], interpolation='nearest')
    axes[i].set_title(f"{class_names[predicted_label]}\n({class_names[true_label]})", fontsize=10)
    axes[i].axis('off')
plt.tight_layout()
plt.show()

# 🧩 Eigene Bilder vorbereiten
def load_and_preprocess_image(path):
    img = Image.open(path).convert('RGB')
    img = img.filter(ImageFilter.SHARPEN)
    img = img.resize((32, 32))
    img_array = np.array(img) / 255.0
    return img_array

# 🐱 Klassifikationsfunktion: Ist das eine Katze?
def is_cat(image_array, model, class_names):
    image_batch = np.expand_dims(image_array, axis=0)
    prediction = model.predict(image_batch)
    predicted_label = prediction[0].argmax()
    confidence = prediction[0][predicted_label]
    label_name = class_names[predicted_label]
    print(f"Vorhersage: {label_name} ({confidence:.2%})")
    return label_name == 'Katze', confidence

# 🧪 Eigene Katzenbilder testen
# Hinweis: Diese Pfade zeigen auf Bilder auf meinem Computer.
# Bitte eigene Bilder verwenden und Pfade entsprechend anpassen.
image_paths = [
    r"C:\Users\Serena\Desktop\Sophia.jpg",
    r"C:\Users\Serena\Desktop\IMG2.jpg"
]

for i, path in enumerate(image_paths, start=1):
    if not os.path.exists(path):
        print(f"⚠️ Bild {i} nicht gefunden: {path}")
        print("Bitte Pfad in 'main.py' anpassen und eigene Bilder verwenden.")
        continue

    img = load_and_preprocess_image(path)
    result, confidence = is_cat(img, model, class_names)
    print(f"Bild {i}: {'Katze ✅' if result else 'Keine Katze ❌'} ({confidence:.2%})")
    plt.imshow(img)
    plt.title(f"Bild {i}: {'Katze' if result else 'Nicht Katze'} ({confidence:.2%})")
    plt.axis('off')
    plt.show()