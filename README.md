# 🌿 MangoLeaf AI: Quantitative Phytopathology, Severity Grading & Epidemiological Decision-Support Platform

[![Python](https://img.shields.io/badge/Python-3.8%20%7C%203.11-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Backend-Flask%203.0-green.svg)](https://flask.palletsprojects.com/)
[![OpenCV](https://img.shields.io/badge/Computer%20Vision-OpenCV%204.9-orange.svg)](https://opencv.org/)
[![TensorFlow](https://img.shields.io/badge/Deep%20Learning-TensorFlow%202.15-red.svg)](https://tensorflow.org/)
[![Phytopathology](https://img.shields.io/badge/Standard-Horsfall--Barratt%20Scale-darkgreen.svg)](#phytopathology-standards)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

An enterprise-grade, precision-agritech platform transforming standard leaf disease classification into **objective mathematical quantification (PLAA %)**, **Explainable AI (Grad-CAM with spatial containment validation)**, **micro-climatic epidemiological risk modeling**, and **tree-canopy-calibrated precision dosage prescriptions**.

---

## 🌟 Key Features & Engineering Innovations

### 1. 📐 Pixel-Level Lesion Planimetry & Horsfall-Barratt Grading
- Multi-space colorimetric segmentation combining **CIE-LAB** and **HSV** to isolate functional chloroplasts from necrotic lesions, powdery mycelium, and gall blisters.
- Computes **Percentage Leaf Area Affected (PLAA %)**:
  $$\text{PLAA} = \left(\frac{\text{Lesion Pixels}}{\max(\text{Leaf Lamina Pixels}, 1)}\right) \times 100$$
- Maps to the standardized **0–5 Horsfall-Barratt Phytopathological Scale** with dynamic visualization (crimson lesion clusters + emerald perimeter).

### 2. 🧠 Explainable AI (Grad-CAM) with Spatial Grounding Containment
- Generates visual saliency activation heatmaps using convolutional gradients with a fail-safe multi-scale chrominance & edge saliency fallback.
- Computes an objective **Spatial Containment Score ($S_{\text{contain}}$)**:
  $$S_{\text{contain}} = \frac{\sum_{(x,y) \in \text{Lamina}} \mathbb{I}(\text{Heatmap}(x,y) \ge P_{70})}{\sum_{(x,y)} \mathbb{I}(\text{Heatmap}(x,y) \ge P_{70})}$$
- Mathematically validates whether neural attention is grounded *inside* the biological leaf rather than hallucinating on background soil, sky, or imaging artifacts.

### 3. 🌦️ Micro-Climatic Spore Risk & Safe Spray-Window Engine
- Ingests orchard telemetry (Temperature, Relative Humidity, Wind Speed, Rain Probability).
- Live weather synchronization via Open-Meteo API with offline fallback.
- Calculates pathogen-specific **Spore Proliferation Risk (0–100 Index & LOW/MODERATE/HIGH/CRITICAL Levels)** based on conidial germination kinetics.
- Evaluates atmospheric **Safe Spray Window** to prevent chemical wash-off ($\text{rain\_prob} \ge 40\%$) and off-target droplet drift ($\text{wind} \ge 18\text{ km/h}$).

### 4. 🧪 Severity-Weighted Canopy Precision Dosage Prescriptions
- Formulates actionable tank-mix prescriptions calibrated for mature mango tree canopies ($15\text{ Liters}$ spray solution per tree):
  $$\text{Total Water} = \text{Tree Count} \times 15\text{ L}$$
  $$\text{Total Chemical} = \text{Total Water} \times \text{Dose Per Liter}$$
- Dynamically selects active agents (Azadirachtin neem bio-fungicides, Carbendazim + Mancozeb, Hexaconazole, or Wettable Sulphur) and mechanical pruning protocols.

### 5. 📜 Clinical Phytopathology Certificate Generation
- Generates printable/downloadable audit certificates complete with specimen metadata, planimetry metrics, spatial grounding stamp, and regulatory field protocols with full `@media print` styling.

---

## 🏗️ Platform Architecture

```
                    ┌────────────────────────────────────────┐
                    │ Raw Orchard Leaf Image + Telemetry     │
                    └───────────────────┬────────────────────┘
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │ Stage 0: Dual Otsu + Canny Lamina Segmentation      │
             └──────────┬───────────────────────────────┬──────────┘
                        │                               │
        ┌───────────────▼──────────────┐ ┌──────────────▼────────────────┐
        │ Engine 1: Lesion Planimetry  │ │ Engine 2: Explainable AI      │
        │ - CIE-LAB + HSV Color Spaces │ │ - Grad-CAM / Edge-Chrom Sali- │
        │ - PLAA % & 0–5 Scale         │ │   ency Map                    │
        │ - Emerald/Crimson Overlay    │ │ - Spatial Containment Score   │
        └───────────────┬──────────────┘ └──────────────┬────────────────┘
                        │                               │
                        └───────────────┬───────────────┘
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │ Engine 3: Microclimate Spore & Spray Window Risk    │
             │ - Open-Meteo live sync / local fallback             │
             │ - Conidial germination index (0–100)               │
             │ - Wind drift & rain wash-off safety gate            │
             └──────────────────────────┬──────────────────────────┘
                                        ▼
             ┌─────────────────────────────────────────────────────┐
             │ Engine 4: Canopy-Calibrated Precision Prescription  │
             │ - Volume = 15 L / tree × tree_count                │
             │ - Severity-scaled active chemical concentration    │
             │ - Agronomical mechanical sanitation protocol       │
             └──────────────────────────┬──────────────────────────┘
                                        ▼
                    ┌────────────────────────────────────────┐
                    │ Backward-Compatible API + Web UI + PDF │
                    └────────────────────────────────────────┘
```

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.8+ (tested on Python 3.8, 3.10, and 3.11)
- Git

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/justdarshan510/mango-leaf-disease-platform.git
   cd mango-leaf-disease-platform
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   # Windows PowerShell
   .\venv\Scripts\Activate.ps1
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the application:**
   ```bash
   python app.py
   ```
   Open your browser to: **`http://localhost:5005`**

---

## 📡 API Specification

### `POST /predict`
Analyzes a leaf image with optional orchard telemetry parameters.

#### Form-Data Parameters:
- `file`: (Binary) Mango leaf image (`.jpg`, `.jpeg`, `.png`)
- `temp`: (Float, optional) Ambient temperature in °C (default: `29.0`)
- `humidity`: (Float, optional) Relative humidity % (default: `82.0`)
- `wind`: (Float, optional) Wind velocity in km/h (default: `11.0`)
- `rain_prob`: (Float, optional) Precipitation probability % (default: `15.0`)
- `tree_count`: (Integer, optional) Number of canopy trees (default: `10`)
- `plot_id`: (String, optional) Orchard plot identifier (default: `"PLOT-A1"`)

#### Response JSON:
```json
{
  "success": true,
  "topPrediction": "Anthracnose",
  "confidence": 0.9782,
  "allPredictions": [
    {"class": "Anthracnose", "probability": 0.9782},
    {"class": "Healthy", "probability": 0.0114},
    {"class": "Die Back", "probability": 0.0051},
    {"class": "Gall Midge", "probability": 0.0031},
    {"class": "Powdery Mildew", "probability": 0.0022}
  ],
  "bbox": [5, 5, 214, 214],
  "polygon": [[5, 5], [219, 5], [219, 219], [5, 219]],
  "severityMetrics": {
    "plaaPercentage": 12.3,
    "severityGrade": 2,
    "severityLabel": "Grade 2 — Mild Infection",
    "lesionPixels": 5420,
    "leafPixels": 44065,
    "lesionMaskBase64": "data:image/png;base64,..."
  },
  "explainability": {
    "gradCamBase64": "data:image/png;base64,...",
    "spatialContainmentScore": 0.85,
    "groundingVerified": true,
    "groundingNote": "85.0% of diagnostic attention is concentrated within the leaf lamina contour."
  },
  "epidemiology": {
    "tempC": 30.5,
    "humidityPct": 85.0,
    "windKmh": 9.0,
    "rainProbPct": 10.0,
    "sporeRiskIndex": 85,
    "sporeRiskLevel": "HIGH",
    "sprayWindowStatus": "OPTIMAL SPRAY WINDOW",
    "sprayWindowSafe": true,
    "advisoryNote": "High humidity & warm temps promote conidial germination. Conditions are optimal for spraying."
  },
  "dosagePrescription": {
    "agentName": "Carbendazim 12% + Mancozeb 63% WP (Saaf)",
    "treatmentType": "Dual Systemic + Contact Fungicide",
    "dosePerLiter": "2.0 g/L",
    "treeCount": 15,
    "totalWaterLiters": 225,
    "totalChemicalAmount": "450.0 g",
    "actionProtocol": "Apply uniform foliar spray covering upper and lower lamina surfaces."
  }
}
```

---

## 📊 Phytopathology Standards & References

1. **Horsfall, J. G., & Barratt, R. W. (1945).** *An improved grading system for measuring plant disease.* Phytopathology, 35(8), 655.
2. **Indian Council of Agricultural Research (ICAR) & Central Institute for Subtropical Horticulture (CISH):** *Integrated Pest & Disease Management for Mango.*
3. **Central Insecticides Board & Registration Committee (CIBRC):** *Approved dosimetric standards for foliar fungicides.*

---

## 👥 Authors & Acknowledgments
- Developed by **Darshan** ([@justdarshan510](https://github.com/justdarshan510))
- Agritech & Quantitative Phytopathology Research Initiative
