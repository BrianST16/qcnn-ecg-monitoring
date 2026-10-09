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
├── src/qcnn_ecg/  # Pipeline de datos, modelos, entrenamiento y evaluación
├── configs/       # Configuraciones de las variantes de la CNN (TOML)
├── results/       # Métricas, curvas y resúmenes de los experimentos
├── modelos_entrenados/ # Pesos del modelo final para el prototipo
├── tests/         # Pruebas (pytest)
├── LICENSE        # Licencia de uso libre MIT
└── README.md      # Descripción principal del proyecto
```

---

## 🚀 Cuadernos de Trabajo

| Cuaderno | Descripción | Enlace |
| :--- | :--- | :---: |
| `EDA_MIT_BIH_ECG.ipynb` | Análisis exploratorio e ingesta de datos usando `wfdb` | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/) |

---

## 🧪 CNN 1-D de referencia (Ciclo 2)

Pipeline reproducible en `src/qcnn_ecg/` (gestionado con [uv](https://docs.astral.sh/uv/)):

```bash
uv sync                                        # entorno (Python 3.12, PyTorch CPU, PennyLane)
uv run python -m qcnn_ecg.preparar_datos       # descarga MIT-BIH y construye el conjunto (~10 s)
uv run python -m qcnn_ecg.experimentos         # ablación con validación cruzada + modelo final (~1.5 h en CPU)
uv run pytest                                  # pruebas
```

- **Datos**: derivación MLII, filtro de 0.5 a 40 Hz, z-score por registro y ventanas de 288 muestras (±0.4 s) alrededor del pico R. Normal = `N`. Anormal = `L R A a J S V E F j e`.
- **Partición inter-paciente** (de Chazal, 2004): DS1 para desarrollo y DS2 para prueba, sin registros con marcapasos. El 202 queda fuera de la prueba porque es la misma persona que el 201. El manifiesto con hashes está en `data/manifiestos/`.
- **Protocolo**: validación cruzada de 5 pliegues por paciente en DS1 para elegir la variante, las épocas y el umbral. Luego, un modelo final con 3 semillas, evaluado una sola vez en prueba.

| Prueba (DS2 sin 202, 3 semillas) | Valor |
| :--- | ---: |
| F1 (clase anormal) | 0.61 ± 0.05 |
| Exactitud | 0.79 ± 0.05 |
| AUC-ROC | 0.84 ± 0.03 |
| Parámetros | 11,265 |
| Latencia por latido (CPU) | < 1 ms |

La sensibilidad varía mucho entre semillas por dos limitaciones conocidas en la elección de época y umbral, documentadas en `results/cnn/resumen.md` y pendientes para el Ciclo 3. Detalles en `results/cnn/resumen.md` y en `notebooks/02_cnn1d_resultados.ipynb`. El modelo para el prototipo está en `modelos_entrenados/cnn1d_final.pt`.

> El notebook EDA original usa TensorFlow; para ejecutarlo localmente: `uv sync --group legacy`.

---

## ⚙️ Metodología de Desarrollo

El desarrollo del proyecto se gestiona bajo la metodología **Shape Up**, organizada en ciclos de 3 semanas con apetito fijo para garantizar la entrega progresiva de componentes funcionales[cite: 3].

---

## 👥 Equipo de Trabajo

* **Juan José Barrientos Salazar** — `juanj.barrientos@udea.edu.co`[cite: 3]
* **Brayan Stiven Tobón Foronda** — `bstiven.tobon@udea.edu.co`[cite: 3]
* **Didian Alejandro Valencia Ruíz** — `didian.valencia@udea.edu.co`[cite: 3]

**Asesor:** Javier Fernando Botia Valderrama[cite: 3]

