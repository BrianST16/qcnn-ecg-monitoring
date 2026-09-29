# Comparación de CNN y QCNN para Clasificación de Señales ECG 🫀⚡

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-QML-purple?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Proyecto Integrador 1 — Universidad de Antioquia (2026-2)**

Prototipo de software para la clasificación de segmentos de electrocardiograma (ECG), comparando el desempeño y costo computacional de una **Red Neuronal Convolucional unidimensional (CNN 1-D)** clásica frente a una **Red Neuronal Convolucional Cuántica (QCNN)** ejecutada en simulador[cite: 3].

---

## 📌 Descripción del Proyecto

El sistema evalúa y contrasta modelos de aprendizaje profundo clásico y cuántico bajo condiciones experimentales equivalentes (misma partición estratificada por paciente)[cite: 3]. 

### Objetivos Principales
* **Preprocesamiento:** Filtrado, normalización y segmentación latido a latido sobre bases públicas de ECG (MIT-BIH / PTB-XL)[cite: 1, 3].
* **Rama Clásica:** Implementación de una CNN 1-D de referencia[cite: 3].
* **Rama Cuántica:** Diseño de una QCNN evaluando codificación por ángulos y amplitud[cite: 3].
* **Métricas de Comparación:** Evaluación en exactitud, precisión, sensibilidad, F1-score y descriptores del circuito cuántico (ancho, profundidad y número de compuertas)[cite: 3].

---

## 🛠️ Arquitectura y Flujo de Trabajo

```text
[ Base de Datos ECG ] ──► [ Preprocesamiento & Filtrado ] ──► [ Partición Estratificada ]
                                                                       │
                                             ┌─────────────────────────┴─────────────────────────┐
                                             ▼                                                   ▼
                                    [ Modelo CNN 1-D ]                                [ Modelo QCNN Simulada ]
                                             │                                                   │
                                             └─────────────────────────┬─────────────────────────┘
                                                                       ▼
                                                          [ Evaluación y Comparación ]
```

---

## 📂 Estructura del Repositorio

```text
.
├── docs/          # Anteproyecto, matriz de antecedentes y documentación general
├── notebooks/     # Cuadernos interactivos (EDA y procesamiento de datos)
├── src/           # Módulos en Python para filtrado, modelos e interfaz
├── LICENSE        # Licencia de uso libre MIT
└── README.md      # Descripción principal del proyecto
```

---

## 🚀 Cuadernos de Trabajo

| Cuaderno | Descripción | Enlace |
| :--- | :--- | :---: |
| `EDA_MIT_BIH_ECG.ipynb` | Análisis exploratorio e ingesta de datos usando `wfdb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) |

---

## ⚙️ Metodología de Desarrollo

El desarrollo del proyecto se gestiona bajo la metodología **Shape Up**, organizada en ciclos de 3 semanas con apetito fijo para garantizar la entrega progresiva de componentes funcionales[cite: 3].

---

## 👥 Equipo de Trabajo

* **Juan José Barrientos Salazar** — `juanj.barrientos@udea.edu.co`[cite: 3]
* **Brayan Stiven Tobón Foronda** — `bstiven.tobon@udea.edu.co`[cite: 3]
* **Didian Alejandro Valencia Ruíz** — `didian.valencia@udea.edu.co`[cite: 3]

**Asesor:** Javier Fernando Botia Valderrama[cite: 3]

