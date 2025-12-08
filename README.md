# NeuroAlzheimer
Kontext des Projekts / Präsentationsrahmen
Dieses Projekt entstand im Rahmen des Moduls Neuroengineering for Human-Centered Interaction an der Universität Duisburg-Essen (UDE), Studiengang Medizintechnik, Vertiefung Telemedizin, im Wintersemester 2025/2026.

Der Vortrag sowie die begleitende Implementierung orientieren sich an dem wissenschaftlichen Artikel:

„Correlation between cognitive brain function and electrical brain activity in patients with Alzheimer’s disease“

## Überblick
NeuroAlzheimer ist ein Python-Projekt zur Analyse von EEG-Signalen mit dem Ziel, potenzielle Biomarker für Alzheimer zu untersuchen. Im Fokus stehen Frequenzveränderungen und Verlust von neuronaler Synchronisation, wie sie in der Forschung häufig mit kognitiven Defiziten und Demenz assoziiert werden.

Das Projekt extrahiert charakteristische EEG-Merkmale und nutzt diese für eine automatisierte Klassifikation zwischen gesunden und erkrankten Probanden.

## Wissenschaftlicher Hintergrund
In zahlreichen Studien wurde gezeigt, dass Alzheimer-Patienten typische Veränderungen im EEG aufweisen, darunter:

- Erhöhte Aktivität in niederfrequenten Bereichen wie Theta
- Verminderte Aktivität im Alpha-Bereich
- Veränderte Peak-Frequenz (Verschiebung zu langsameren Wellen)
- Allgemeine Reduktion der kortikalen Konnektivität

Diese Merkmale werden in diesem Projekt ausgewertet, visualisiert und zur maschinellen Klassifikation genutzt.

## Features  

- Laden und Vorverarbeiten von EEG-Daten  
- Berechnung von Bandpower in verschiedenen Frequenzbändern  
- Extraktion der Peak Frequency im Alpha-Band  
- Visualisierung der Ergebnisse (Histogramme, Balkendiagramme, Scatterplots)  
- Einfaches Machine Learning zur Klassifikation (z. B. gesund vs. Alzheimer)  
- Ausgabe von Performance-Metriken (Accuracy, Konfusionsmatrix, Precision / Recall / F1-Score)  

## Voraussetzungen  

- Python 3.10 (oder kompatibel)  
- Benötigte Python-Pakete: numpy, scipy, matplotlib, pandas, scikit-learn (oder wie in requirements.txt angegeben)  
- EEG-Daten im passenden Format (je nachdem, wie dein Loader implementiert ist)

## Projektstruktur

```text
├── data/
│   ├── participants_info.xlsx
│   ├── alzheimer/
│   │   ├── sub-001_task-eyesclosed_eeg.set
│   │   └── ...
│   ├── healthy/
│       ├── sub-037_task-eyesclosed_eeg.set
│       └── ...
├── paper/
├── code.py                # Hauptskript
├── presentation.pptx      # Präsentation (UDE)
└── README.md              
