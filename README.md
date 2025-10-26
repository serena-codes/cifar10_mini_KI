# 🐱 CIFAR10 – Ist das eine Katze?

Ein kleines Deep-Learning-Projekt mit CIFAR-10 und TensorFlow, das Bilder in 10 Klassen klassifiziert und prüft, ob die Vorhersage „Katze“ lautet. 

## 🔍 Funktionen

- Trainiert ein CNN-Modell auf dem CIFAR-10 Datensatz
- Zeigt Vorhersagen auf Testbildern
- Neue Funktion: „Ist das eine Katze?“ – prüft eigene Bilder
- Visualisiert Ergebnisse mit Matplotlib

## 🛠️ Technologien

Dieses Projekt wurde mit folgenden Technologien und Bibliotheken umgesetzt:

- **Python 3.8+** – Programmiersprache
- **TensorFlow** – Deep Learning Framework für das CNN-Modell
- **NumPy** – Numerische Berechnungen und Array-Verarbeitung
- **Matplotlib** – Visualisierung von Bildern und Vorhersagen
- **Pillow (PIL)** – Bildverarbeitung für eigene Fotos
- PyCharm – Entwicklungsumgebung 
- CIFAR-10 Dataset – öffentliches Bilddatenset für Klassifikation

## 🧠 Modellarchitektur

- 2 Convolutional Layers mit MaxPooling
- Dropout gegen Overfitting
- Dense Layer mit Softmax-Ausgabe (10 Klassen)

## 🐾 Eigene Bilder testen

Du kannst eigene Bilder verwenden, z.B.:

```python
image_paths = [
    "pfad/zur/deiner_katze1.jpg",
    "pfad/zur/deiner_katze2.jpg"
]
```
Die Bilder werden automatisch auf 32×32 Pixel skaliert und leicht nachgeschärft – du musst sie nicht vorher anpassen.

## ⚠️ Wichtig: 
Die oben genannten Pfade sind Platzhalter. Bitte ersetze sie durch eigene Bildpfade auf deinem Computer. 
Wenn ein Bild nicht gefunden wird, gibt das Programm eine freundliche Warnung aus – es stürzt nicht ab.

## 🐾 Realitätscheck – Warum erkennt das Modell meine Katze nicht?

Beim Test mit echten Katzenbildern (z.B. Katze auf dem Rücken) habe ich gelernt, dass Modelle Grenzen haben. 
Mein Modell wurde auf dem CIFAR-10-Datensatz trainiert – und der ist sehr speziell:

- **CIFAR-10 ist stark vereinfacht**: Die „Katze“-Bilder darin sind winzige, klare Fotos mit typischer Pose.
- **Eigene Bilder sind komplexer**: Sie sind hochauflösend, realistisch und zeigen Katzen in ungewöhnlichen Perspektiven.
- **Beispiel Katze**: Rückenlage + Perspektive + Fellmuster = für das Modell: „Was ist das?! Vielleicht ein Frosch?“ 🐸

### 🛠️ Was ich demnächst ausprobieren möchte

1. **📚 Transfer Learning mit einem größeren Modell**  
   Statt CIFAR-10 kann man ein vortrainiertes Modell wie MobileNet oder ResNet verwenden – die erkennen echte Katzen deutlich besser.

2. **🐱 Eigenes Mini-Dataset erstellen**  
   Ich könnte ein paar Bilder meiner Katze in verschiedenen Posen sammeln und ein kleines Modell nur für „Katze vs. Nicht-Katze“ trainieren.

3. **🖼️ Bilder vorher zuschneiden oder zentrieren**  
   Wenn man den Hintergrund entfernt oder die Katze zentriert, steigt die Chance, dass das Modell sie korrekt erkennt.

## 🛠️ Voraussetzungen

- Python 3.8+
- TensorFlow
- Pillow
- Matplotlib
- NumPy

Installiere alles mit:

```
pip install tensorflow pillow matplotlib numpy
```
## Starten
```
python main.py
```

## 👩‍💻 Autor

Erstellt von Serena – mit viel Lernfreude 😄

## 📄 Lizenz

Dieses Projekt kann frei verwendet und angepasst werden – für Lernzwecke, Spaß und persönliche Weiterentwicklung.

