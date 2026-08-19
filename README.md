# NeuroAlzheimer
This project was developed as part of the "Neuroengineering for Human-Centered Interaction" module at the University of Duisburg-Essen (UDE)—within the Medical Engineering degree program, specializing in Telemedicine—during the 2025/2026 winter semester.

The presentation and the accompanying implementation are based on the following scientific article:

"Correlation between cognitive brain function and electrical brain activity in patients with Alzheimer’s disease"

## Overview
NeuroAlzheimer is a Python-based project designed to analyze EEG signals with the aim of investigating potential biomarkers for Alzheimer's disease. It focuses on frequency changes and the loss of neuronal synchronization—phenomena frequently associated in research with cognitive deficits and dementia.

The project extracts characteristic EEG features and uses them to automatically classify subjects as either healthy or affected by the disease.

## Scientific Background
Numerous studies have demonstrated that Alzheimer's patients exhibit typical EEG alterations, including:

- Increased activity in low-frequency bands (such as the theta band)
- Decreased activity in the alpha band
- Altered peak frequency (a shift toward slower waves)
- A general reduction in cortical connectivity

In this project, these features are analyzed, visualized, and utilized for automated classification. ## Features

- Loading and preprocessing of EEG data
- Calculation of band power across various frequency bands
- Extraction of peak frequency in the alpha band
- Visualization of results (histograms, bar charts, scatter plots)
- Basic machine learning for classification (e.g., healthy vs. Alzheimer's)
- Output of performance metrics (accuracy, confusion matrix, precision/recall/F1-score)

## Prerequisites

- Python 3.10 (or compatible)
- Required Python packages: numpy, scipy, matplotlib, pandas, scikit-learn
- EEG data in the appropriate format (depending on the loader implementation)

## Project Structure

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
├── code.py                # Main script
├── presentation.pptx      # Presentation (UDE)
└── README.md
```
