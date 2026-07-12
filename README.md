# DNA Strand Displacement Prediction

Machine learning approach for predicting toehold-mediated DNA strand displacement kinetics using sequence-based and thermodynamic features.

University of Newcastle CSC3094 Dissertation Project.

[📄 View dissertation](./report/dissertation.pdf) [💻 View source](./code)

---

## Overview

This project explores the prediction of DNA strand displacement reaction rates using machine learning and computational biology approaches.

The pipeline combines:

- DNA language model embeddings
- Thermodynamic nearest-neighbour features
- Sequence availability features
- Regression-based prediction models

The aim is to improve prediction of molecular reaction behaviour and support the design of DNA-based computing systems.

---

## Methodology

The project investigates the combination of biological sequence representations and traditional thermodynamic modelling.

Key components include:

- DNA-BERT sequence embeddings
- Nearest-neighbour thermodynamic analysis
- Feature engineering from molecular properties
- Machine learning model evaluation
- Reproducible experimental workflows

---

## Results

The final model combines transformer-based sequence embeddings with thermodynamic and availability features.

Performance:

- R²: 0.702
- External ranking Spearman correlation: 0.745
- 64.5% of predictions within 2× of measured rates

The results demonstrate that combining learned sequence representations with domain-specific biological features improves prediction of DNA strand displacement kinetics.

---

## Report

The full dissertation is available here:

[📄 Read dissertation](./report)
