# Resultados de la CNN 1-D v2 (rasgos RR, media centrada y ensamble)

- Fecha: 2026-11-16 18:35
- Equipo: Intel64 Family 6 Model 140 Stepping 1, GenuineIntel, PyTorch 2.14.1+cpu
- Conjunto: `mitdb_binario_v2` (ver `data/manifiestos/`)
- Cómputo de esta sesión: **47.4 min**. Semillas: 0, 1, 2.

## Protocolo

1. Validación cruzada de 5 pliegues por paciente en DS1; la parada temprana usa el F1 suavizado con media **centrada** (t-1, t, t+1).
2. Las predicciones fuera de pliegue (OOF) fijan el umbral que maximiza el F1.
3. El modelo final es el **ensamble de los 5 modelos de pliegue** con ese umbral (el umbral se calculó con los mismos modelos a los que se aplica). Una sola evaluación en prueba.
4. Variante elegida por F1 OOF medio: **base**.

## Comparación (media ± desviación de las semillas)

| Variante | F1 OOF | F1 prueba | AUC-ROC prueba | AUC-PR prueba | Sensibilidad prueba | Acierto en A |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| base **(elegida)** | 0.6378 ± 0.0173 | 0.5866 ± 0.0583 | 0.8511 ± 0.0095 | 0.6880 ± 0.0185 | 0.5456 ± 0.1743 | 0.1120 ± 0.1246 |
| base_rr | 0.6223 ± 0.0378 | 0.5955 ± 0.0553 | 0.8586 ± 0.0134 | 0.6985 ± 0.0271 | 0.5210 ± 0.1122 | 0.0757 ± 0.0462 |
| v1 `base` (results/cnn/) | 0.7320 | 0.6058 | 0.8397 | 0.6780 | 0.6097 | 0.3002 |

Prueba = DS2 sin el registro 202. La fila v1 es la de `results/cnn/resumen.md`.

## Detalle por semilla

variante,semilla,umbral,oof_exactitud,oof_precision,oof_sensibilidad,oof_especificidad,oof_f1,oof_f1_macro,oof_auc_roc,oof_auc_pr,prueba_exactitud,prueba_precision,prueba_sensibilidad,prueba_especificidad,prueba_f1,prueba_f1_macro,prueba_auc_roc,prueba_auc_pr,acierto_A
base,0,0.7068,0.8086,0.6241,0.6133,0.8748,0.6187,0.7455,0.837,0.6777,0.8074,0.8167,0.3934,0.9661,0.531,0.7049,0.8541,0.7007,0.0294
base,1,0.4742,0.8126,0.615,0.6945,0.8527,0.6523,0.7621,0.8004,0.6213,0.7975,0.6808,0.5076,0.9087,0.5816,0.724,0.8405,0.6668,0.0512
base,2,0.5014,0.7912,0.567,0.7408,0.8083,0.6423,0.7474,0.8018,0.5776,0.7776,0.5777,0.7358,0.7937,0.6472,0.7424,0.8587,0.6964,0.2553
base_rr,0,0.7072,0.7864,0.5647,0.6802,0.8223,0.6171,0.7345,0.8256,0.639,0.8099,0.824,0.3999,0.9672,0.5384,0.7094,0.8705,0.7165,0.03
base_rr,1,0.5276,0.8149,0.615,0.7178,0.8478,0.6624,0.7675,0.8031,0.6364,0.799,0.6702,0.5418,0.8977,0.5992,0.7325,0.8441,0.6674,0.0747
base_rr,2,0.5694,0.7516,0.5068,0.6987,0.7696,0.5875,0.7049,0.7876,0.548,0.8135,0.6788,0.6213,0.8872,0.6488,0.7609,0.8614,0.7116,0.1224

