# Bilderkenner

Ein Lernprojekt zur Bildklassifikation mit Deep Learning.

---

## Was macht dieses Projekt?

Dieses Projekt trainiert ein **neuronales Netz**, das Bilder erkennt und in 10 Kategorien einteilt:

| Flugzeug | Auto | Vogel | Katze | Hirsch |
|----------|------|-------|-------|--------|
| Hund | Frosch | Pferd | Schiff | LKW |

Das Modell lernt anhand von 60.000 Bildern aus dem CIFAR-10 Datensatz.

---

## Schnellstart

### 1. Repository holen

**Option A: Klonen (wenn du Schreibrechte hast)**
```bash
git clone https://github.com/SteffZeroEx/bilderkenner-projekt.git
cd bilderkenner-projekt
```

**Option B: Forken (eigene Kopie erstellen)**
1. Klicke oben rechts auf "Fork"
2. Dann:
```bash
git clone https://github.com/DEIN-USERNAME/bilderkenner-projekt.git
cd bilderkenner-projekt
```

### 2. Python-Umgebung einrichten

```bash
# Virtuelle Umgebung erstellen
python -m venv venv

# Aktivieren
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows

# Pakete installieren
pip install -r requirements.txt
```

### 3. Testen ob alles funktioniert

```bash
python test_setup.py
```

---

## Projektstruktur

```
bilderkenner-projekt/
│
├── data/                 # Datenverarbeitung
│   ├── dataset.py        # Laedt CIFAR-10 Bilder
│   ├── dataloader.py     # Bereitet Daten vor
│   └── transforms.py     # Bildtransformationen
│
├── models/               # Das neuronale Netz
│   ├── cnn.py            # CNN Architektur
│   ├── train.py          # Training starten
│   └── predict.py        # Vorhersagen machen
│
├── api/                  # REST API
│   ├── main.py           # Server starten
│   ├── routes.py         # API Endpunkte
│   └── schemas.py        # Datenformate
│
├── tests/                # Automatische Tests
└── saved_models/         # Gespeicherte Modelle
```

---

## Training starten

```bash
python -m models.train
```

Das Training dauert je nach Hardware einige Minuten. Das beste Modell wird automatisch gespeichert.

---

## API starten

```bash
uvicorn api.main:app --reload
```

Dann im Browser oeffnen: http://localhost:8000/docs

Dort kannst du Bilder hochladen und klassifizieren lassen!

---

## Tests ausfuehren

```bash
pytest tests/ -v
```

---

## Git Workflow fuer Anfaenger

### Aenderungen speichern und hochladen

```bash
# 1. Status pruefen - was hat sich geaendert?
git status

# 2. Alle Aenderungen zum Commit hinzufuegen
git add .

# 3. Commit erstellen (mit Beschreibung)
git commit -m "Beschreibung was du geaendert hast"

# 4. Auf GitHub hochladen
git push
```

### Neueste Version holen

```bash
git pull
```

### Neuen Branch erstellen (fuer Features)

```bash
# Branch erstellen und wechseln
git checkout -b mein-feature

# Arbeiten, committen...

# Zurueck zu main
git checkout main

# Branch mergen
git merge mein-feature
```

---

## Technologien

| Bibliothek | Verwendung |
|------------|------------|
| PyTorch | Deep Learning Framework |
| torchvision | Bildverarbeitung |
| FastAPI | REST API |
| pytest | Testing |

---

## Code Beispiel

```python
from data.dataset import CIFAR10Dataset
from models.cnn import get_model

# Dataset laden
dataset = CIFAR10Dataset()
print(f"Klassen: {dataset.classes}")

# Modell erstellen
model = get_model()
print(model)
```

---

## Mitwirkende

- SteffZeroEx
- theunder-fITler
