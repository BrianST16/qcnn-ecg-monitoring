# Resultados de la CNN 1-D (Ciclo 2)

- Fecha: 2026-10-08 21:23
- Equipo: Intel64 Family 6 Model 142 Stepping 12, GenuineIntel, 4 hilos, PyTorch 2.14.1+cpu (CPU)
- Conjunto: `mitdb_binario_v1` (ver `data/manifiestos/`)
- Tiempo de cómputo de este experimento: **102.4 min** (unos 27 min adicionales se usaron en la corrida previa con un solo pliegue; presupuesto total 150 min)
- Semillas finales: **0, 1, 2**. Las 3 semillas cupieron en el presupuesto.

## Protocolo

1. Validación cruzada de 5 pliegues por paciente en DS1 (22 registros), semilla 0.
   Cada pliegue usa parada temprana con el F1 suavizado (media de 3 épocas).
2. Las predicciones fuera de pliegue (OOF) de los 22 pacientes miden cada variante y
   fijan el umbral que maximiza el F1 de la clase anormal.
3. El modelo final se entrena con todo DS1 durante la mediana de las mejores épocas de
   los pliegues, con varias semillas, y se evalúa una sola vez en prueba.

Pliegues de pacientes en DS1:

- Pliegue 0: 119, 203, 208, 215, 223 (13,879 latidos)
- Pliegue 1: 112, 124, 201, 209 (9,122 latidos)
- Pliegue 2: 101, 109, 115, 230 (8,599 latidos)
- Pliegue 3: 106, 108, 116, 205, 207 (10,714 latidos)
- Pliegue 4: 114, 118, 122, 220 (8,676 latidos)

## Ablación (validación cruzada en DS1, semilla 0)

| Variante | Parámetros | Min | Épocas final | F1 OOF | Sensibilidad | Especificidad | AUC-PR OOF | F1 por pliegue |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base **(elegida)** | 11,265 | 11.8 | 3 | 0.7320 | 0.7361 | 0.9068 | 0.6646 | 0.744 ± 0.224 |
| kiranyaz | 9,605 | 8.4 | 3 | 0.6968 | 0.7942 | 0.8355 | 0.6647 | 0.758 ± 0.219 |
| flatten | 82,945 | 17.5 | 6 | 0.6931 | 0.8395 | 0.8024 | 0.7017 | 0.780 ± 0.181 |
| dropout_05 | 11,265 | 9.0 | 3 | 0.6447 | 0.6793 | 0.8550 | 0.5984 | 0.715 ± 0.197 |
| kernels_grandes | 22,657 | 20.3 | 5 | 0.6257 | 0.7794 | 0.7588 | 0.5687 | 0.704 ± 0.171 |
| filtros_x2 | 39,873 | 32.8 | 4 | 0.6192 | 0.6087 | 0.8789 | 0.5398 | 0.695 ± 0.172 |

## Prueba: variante `base`, media ± desviación de 3 semillas

Umbral fijado con la CV en DS1: **0.6150**. Épocas del modelo final: **3**. La prueba se evaluó una sola vez por semilla.

| Métrica | DS2 sin 202 (principal) | DS2 con 202 (comparable con la literatura) |
| :--- | ---: | ---: |
| exactitud | 0.7869 ± 0.0496 | 0.7899 ± 0.0551 |
| precision | 0.6854 ± 0.1622 | 0.6785 ± 0.1682 |
| sensibilidad | 0.6097 ± 0.2291 | 0.6096 ± 0.2292 |
| especificidad | 0.8548 ± 0.1542 | 0.8555 ± 0.1563 |
| f1 | 0.6058 ± 0.0469 | 0.6012 ± 0.0434 |
| f1_macro | 0.7271 ± 0.0225 | 0.7266 ± 0.0256 |
| auc_roc | 0.8397 ± 0.0271 | 0.8409 ± 0.0260 |
| auc_pr | 0.6780 ± 0.0194 | 0.6728 ± 0.0179 |

### Matriz de confusión en prueba por semilla ([[VN, FP], [FN, VP]])

- Semilla 0: `[[33203, 1163], [7712, 5470]]`
- Semilla 1: `[[23330, 11036], [1817, 11365]]`
- Semilla 2: `[[31597, 2769], [5906, 7276]]`

### Acierto por tipo de latido

Para `N` equivale a la especificidad; para los demás, a su sensibilidad.

| Símbolo | Latidos en prueba | Acierto en prueba | Acierto OOF en DS1 |
| :--- | ---: | ---: | ---: |
| N | 34,366 | 0.8548 ± 0.1542 | 0.9068 |
| L | 4,124 | 0.4044 ± 0.3311 | 0.9759 |
| R | 3,475 | 0.7855 ± 0.1792 | 0.5217 |
| V | 3,200 | 0.9019 ± 0.0690 | 0.8947 |
| A | 1,700 | 0.3002 ± 0.4202 | 0.2852 |
| F | 387 | 0.3979 ± 0.2682 | 0.0870 |
| j | 213 | 0.3599 ± 0.1583 | 0.0625 |
| J | 51 | 0.0392 ± 0.0340 | 0.0000 |
| a | 31 | 0.3441 ± 0.2781 | 0.3800 |
| E | 1 | 0.0000 ± 0.0000 | 0.7143 |

### Costo

- Parámetros: 11,265
- Entrenamiento final por semilla: 0.87 ± 0.01 min
- Latencia por latido (lote de 1): mediana 0.892 ± 0.149 ms; p95 2.388 ± 1.835 ms
- Throughput (lotes de 256): 9463 ± 1725 latidos/s

Modelo para el prototipo: semilla 0 (por convención; las semillas finales no tienen validación propia para elegir entre ellas), copiado a `modelos_entrenados/cnn1d_final.pt`.

## Limitaciones conocidas (pendientes para el Ciclo 3)

1. **Media móvil no centrada.** La época se elige con la media del F1 de las épocas t-2, t-1 y t, pero se guarda el estado de la época t: las épocas anteriores le dan crédito. En 4 de 5 pliegues de `base` la mejor época resultó ser la 3, por lo que el modelo final entrena solo 3 épocas. Corrección: media centrada (t-1, t, t+1).
2. **Umbral calculado con otros modelos.** El umbral sale de las predicciones de los 5 modelos de la validación cruzada (con su propio punto de parada y reducción de tasa de aprendizaje), pero se aplica a un modelo final distinto, entrenado con todo DS1 y otra semilla. Por eso la sensibilidad en prueba varía mucho entre semillas (0.42 a 0.86). Corrección: usar como modelo final el ensamble de los 5 modelos de la CV.
3. La CV de la ablación usa una sola semilla; el filtro `filtfilt` no es causal.

La corrida anterior con un único pliegue de validación (5 pacientes) está en `results/cnn_pliegue_unico/`: allí el umbral elegido (≈ 0.997) no se trasladó a la prueba (F1 0.29 con AUC-ROC 0.90), lo que motivó este protocolo.
