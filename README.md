# RampSense

Ultra-short-term photovoltaic and wind power ramp-event forecasting using LSTM-based deep learning and hybrid optimization.

## Overview

RampSense is a semester research project based on the IEEE Access 2026 paper:

> **A Novel Combined Objective Hybrid Framework for Ultra-short-term Ramp Event Forecasting in Photovoltaic and Wind Power Generation**

The project investigates ultra-short-term renewable power forecasting with a particular focus on **ramp events**.

A conventional forecasting model can achieve relatively low average power-prediction error while still missing sudden changes in renewable generation. Therefore, this project evaluates both:

1. **Power forecasting accuracy**
2. **Ramp-event detection performance**

The project first establishes a baseline based on the methodology of the reference paper and then investigates improvements proposed for this research project.

---

# Research Objectives

The project has two main stages.

## Stage 1 — Baseline / Reference Framework

The baseline investigates:

- LSTM-based PV power forecasting
- Multi-step forecasting
- Ramp-event identification
- Combined forecasting and ramp-event objectives
- SAMURAI optimization
- Particle Swarm Optimization (PSO)
- ORE/PRE-based ramp evaluation

## Stage 2 — Proposed Framework

The proposed research direction investigates:

- Soft-F1 instead of discrete F1
- False-negative penalty
- Adam-based optimization
- Multi-season and multi-site evaluation
- Transformer-based forecasting
- Comparison between the reference and proposed optimization approaches

The proposed framework will be implemented only after establishing the reference baseline.

---

# Reference Paper

**Kostoglou, Styliani E., et al.**

> *A Novel Combined Objective Hybrid Framework for Ultra-short-term Ramp Event Forecasting in Photovoltaic and Wind Power Generation.*

IEEE Access, 2026.

The reference framework combines forecasting error and ramp-event performance through a combined objective involving MSE and F1.

---

# Project Pipeline

The current project pipeline is:

```text
Raw Renewable Energy Data
          │
          ▼
     Preprocessing
          │
          ▼
     Ramp Detection
          │
          ▼
    Sequence Creation
          │
          ▼
 Chronological Train/Val/Test
          │
          ▼
       Scaling
          │
          ▼
     LSTM Baseline
          │
          ▼
 Multi-step LSTM Forecasting
          │
          ▼
       ORE / PRE
          │
          ▼
    SAMURAI → PSO
          │
          ▼
 Reference Framework
          │
          ▼
 Proposed Soft-F1 +
 False-Negative Penalty
