# Resultados de la CNN 1-D (Ciclo 2)

- Fecha: 2026-10-08 19:15
- Equipo: Intel64 Family 6 Model 142 Stepping 12, GenuineIntel, 4 hilos, PyTorch 2.14.1+cpu (CPU)
- Conjunto: `mitdb_binario_v1` (ver `data/manifiestos/`)
- Tiempo total de cómputo: **19.2 min** (presupuesto 150 min)
- Semillas finales: **0, 1, 2**. Las 3 semillas cupieron en el presupuesto.

## Ablación (semilla 0, solo validación)

| Variante | Parámetros | Épocas (mejor) | Min | F1 | Sensibilidad | Especificidad | AUC-PR |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| kiranyaz **(elegida)** | 9,605 | 8 (3) | 1.3 | 0.7952 | 0.7756 | 0.9439 | 0.8478 |
| flatten | 82,945 | 18 (13) | 4.3 | 0.7786 | 0.8406 | 0.8980 | 0.8169 |
| dropout_05 | 11,265 | 8 (3) | 1.6 | 0.7365 | 0.9511 | 0.7978 | 0.6517 |
| kernels_grandes | 22,657 | 9 (4) | 2.3 | 0.7354 | 0.9580 | 0.7928 | 0.5985 |
| base | 11,265 | 8 (3) | 1.6 | 0.7333 | 0.9500 | 0.7948 | 0.6627 |
| filtros_x2 | 39,873 | 8 (3) | 3.7 | 0.7211 | 0.9473 | 0.7823 | 0.6154 |

## Prueba: variante `kiranyaz`, media ± desviación de 3 semillas

Umbral fijado en validación; la prueba se evaluó una sola vez por semilla.

| Métrica | DS2 sin 202 (principal) | DS2 con 202 (comparable con la literatura) |
| :--- | ---: | ---: |
| exactitud | 0.7682 ± 0.0025 | 0.7766 ± 0.0024 |
| precision | 0.9654 ± 0.0090 | 0.9654 ± 0.0090 |
| sensibilidad | 0.1699 ± 0.0103 | 0.1690 ± 0.0103 |
| especificidad | 0.9976 ± 0.0007 | 0.9978 ± 0.0007 |
| f1 | 0.2889 ± 0.0148 | 0.2875 ± 0.0148 |
| f1_macro | 0.5752 ± 0.0080 | 0.5775 ± 0.0080 |
| auc_roc | 0.8974 ± 0.0195 | 0.9005 ± 0.0186 |
| auc_pr | 0.7752 ± 0.0302 | 0.7740 ± 0.0304 |

### Matriz de confusión en prueba por semilla ([[VN, FP], [FN, VP]])

- Semilla 0: `[[34290, 76], [10853, 2329]]`
- Semilla 1: `[[34307, 59], [11099, 2083]]`
- Semilla 2: `[[34258, 108], [10874, 2308]]`

### Acierto por tipo de latido en prueba

Para `N` equivale a la especificidad; para los demás, a su sensibilidad.

| Símbolo | Latidos | Acierto (media ± desv.) |
| :--- | ---: | ---: |
| N | 34,366 | 0.9976 ± 0.0007 |
| L | 4,124 | 0.0032 ± 0.0038 |
| R | 3,475 | 0.0000 ± 0.0000 |
| V | 3,200 | 0.6933 ± 0.0392 |
| A | 1,700 | 0.0027 ± 0.0024 |
| F | 387 | 0.0086 ± 0.0015 |
| j | 213 | 0.0000 ± 0.0000 |
| J | 51 | 0.0000 ± 0.0000 |
| a | 31 | 0.0000 ± 0.0000 |
| E | 1 | 0.0000 ± 0.0000 |

### Costo

- Parámetros: 9,605
- Entrenamiento por semilla: 1.6641 ± 0.6174 min
- Latencia por latido (lote de 1): mediana 0.4458 ± 0.0417 ms; p95 0.5861 ± 0.0721 ms
- Throughput (lotes de 256): 11252.8586 ± 489.5471 latidos/s

Modelo final (mejor semilla por F1 en validación, no por prueba): semilla 1, copiado a `modelos_entrenados/cnn1d_final.pt`.
