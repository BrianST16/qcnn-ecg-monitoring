# Plan de implementación de la QCNN en Qiskit (Ciclo 3)

**Objetivo:** una QCNN simulada que clasifique latidos ECG como NORMAL/ANORMAL bajo **las mismas condiciones experimentales** que la CNN 1-D, para compararlas de forma justa y conectarla al backend (`resultados.qcnn` del contrato).

## 1. Punto de partida (lo que ya existe y se reutiliza)

| Pieza | Dónde | Qué aporta a la QCNN |
| :--- | :--- | :--- |
| Pipeline de preprocesamiento | `src/qcnn_ecg/pipeline.py` | `procesar_senal` → ventanas (n, 288) + rasgos RR (n, 4). Es el único camino de datos: entrenamiento, backend y QCNN lo comparten. |
| Conjunto y partición | `conjunto.cargar_conjunto()`, `particion.pliegues_ds1` | Mismo DS1/DS2, mismos 5 pliegues por paciente. |
| Protocolo | `validacion_cruzada.ejecutar_cv` | CV por paciente, umbral OOF, ensamble de los 5 modelos de pliegue, evaluación única en prueba. |
| Métricas | `evaluacion.py` | `calcular_metricas`, `acierto_por_simbolo`, `medir_inferencia`. |
| Referencia | `results/cnn_v2/resumen.md` | F1, AUC-ROC, AUC-PR y acierto por símbolo de la CNN a superar o igualar con menos parámetros. |
| Integración backend | `backend/app/modelos/qcnn.py` | `ClasificadorQCNN(ruta)` con dos TODO (cargar y `probabilidades(ventanas, rr)`); `QCNN_MODO=real`. |

## 2. Qué se va a usar

| Paquete | Para qué | Nota |
| :--- | :--- | :--- |
| `qiskit` (2.x) | Circuitos (`QuantumCircuit`, `ParameterVector`) | Versión exacta a fijar con `uv add` y verificar el resolver. |
| `qiskit-machine-learning` (≥ 0.9) | `EstimatorQNN` + `TorchConnector` | Desde la 0.9 es compatible con Qiskit 2.x y **solo** con primitivas V2. |
| `qiskit` `StatevectorEstimator` (`qiskit.primitives`) | Simulación exacta, sin ruido | Es la opción base: determinista y suficiente para 8 qubits. |
| `qiskit-aer` (opcional) | `EstimatorV2` de Aer, y simulación con ruido (`NoiseModel`) | Solo para el experimento de robustez al ruido. |
| `torch` (ya instalado) | Bucle de entrenamiento | `TorchConnector` convierte la QNN en un `nn.Module`: se reutiliza `entrenar_modelo`, `CargadorLotes`, `Ensamble`. |

`pennylane` queda en `pyproject.toml` desde el Ciclo 2 y no se usará: se retira cuando se agregue Qiskit.

Antes de codificar hay que confirmar contra la documentación vigente los imports exactos (`qiskit_machine_learning.neural_networks.EstimatorQNN`, `qiskit_machine_learning.connectors.TorchConnector`) y el parámetro `input_gradients=True`, necesario para que el gradiente atraviese un modelo clásico previo al circuito.

## 3. Arquitectura propuesta (híbrida, 8 qubits)

```
ventana (288) ──► front-end clásico ──► 8 ángulos ──► circuito QCNN (8 qubits) ──► <Z> ──► logit
rasgos RR (4) ───────────────────────────────────────────────────────────────────► capa lineal final
```

1. **Reducción 288 → 8.** Un ECG de 288 muestras no cabe en un circuito de pocos qubits. Se evalúan dos opciones, en este orden:
   - **A. Front-end convolucional pequeño** (1–2 `Conv1d` + pooling global + `Linear(→8)`), entrenado junto con el circuito. Es lo habitual en modelos híbridos y mantiene la comparación justa, pero el front-end clásico hace parte del trabajo: se reporta su número de parámetros por separado.
   - **B. PCA a 8 componentes** ajustada solo con el entrenamiento de cada pliegue (sin fuga de información). Más "puramente cuántica", aunque pierde forma de onda.
2. **Codificación.** `RY(x_i)` por qubit con los 8 valores reescalados a [0, π] (codificación por ángulo; un dato por qubit, profundidad 1).
3. **Circuito QCNN** (Cong, Choi y Lukin, 2019), con pesos compartidos por capa:
   - Capa de convolución: bloque de 2 qubits (`RY`/`RZ` parametrizados + `CNOT`) aplicado a pares vecinos, en dos pasadas (pares pares e impares).
   - Capa de pooling: bloque que mide/descarta un qubit de cada par, dejando 8 → 4 → 2 → 1.
   - 3 pasos de convolución + pooling. Parámetros del circuito: del orden de decenas.
4. **Salida.** Valor esperado de `Z` en el qubit restante (en [-1, 1]), escalado por una capa lineal de 1 parámetro y sesgo hasta un logit. Así `BCEWithLogitsLoss` con `pos_weight` y `predecir_probabilidades` funcionan sin cambios.
5. **Rasgos RR.** Entran por la vía clásica (una `Linear` de 4 → 1 sumada al logit), igual que en la CNN. No se codifican en el circuito: costaría qubits sin aportar a la comparación.

Todo se instancia como `nn.Module` con `forward(x, rr=None)`, la misma firma que `CNN1D` y `Ensamble`, de modo que `evaluacion.py` y el backend no necesitan cambios.

## 4. Entrenamiento y evaluación

- **Protocolo idéntico a la CNN:** CV de 5 pliegues por paciente en DS1, umbral que maximiza el F1 en OOF, ensamble de los 5 modelos, una sola evaluación en DS2 sin el 202 (y aparte con 202).
- **Costo.** Simular un lote de 256 latidos con `StatevectorEstimator` es órdenes de magnitud más lento que la CNN (la CNN tarda unos 0.9 min por entrenamiento completo). Mitigaciones, en orden:
  1. Probar primero con un 10 % del entrenamiento (`--rapido`) para medir segundos por época.
  2. Submuestreo balanceado del entrenamiento de cada pliegue (p. ej. 8 000 latidos) si hace falta; se documenta como desviación respecto a la CNN.
  3. Menos semillas (1 en lugar de 3) para la ablación; 3 solo para el modelo final.
- **Gradientes.** `TorchConnector` usa el gradiente por parameter-shift: el costo crece con el número de parámetros del circuito. Se mantiene el circuito pequeño a propósito.
- **Experimentos de ablación** (cada uno con la misma CV): front-end A frente a B; 2 frente a 3 capas de convolución/pooling; con y sin RR.
- **Ruido (opcional, al final).** Repetir solo la inferencia del modelo final con `qiskit-aer` y un `NoiseModel` de un backend falso (`FakeBackend` de `qiskit_ibm_runtime.fake_provider`) para reportar la degradación. No es parte de la comparación principal.

## 5. Métricas de comparación con la CNN

Mismas que `results/cnn_v2/resumen.md`: F1, sensibilidad, especificidad, AUC-ROC, AUC-PR, acierto por símbolo (en especial `A`, `L`, `V`), media ± desviación entre semillas. Además: **parámetros** (clásicos y cuánticos por separado), **tiempo de entrenamiento** y **tiempo de inferencia por latido** (`medir_inferencia`). La conclusión debe ser honesta: una QCNN simulada difícilmente será más rápida que la CNN; la pregunta es si iguala su desempeño con menos parámetros.

## 6. Integración en el backend

1. Guardar el ensamble como `modelos_entrenados/qcnn_final.pt` con `{"estados", "config_modelo", "umbral"}` (mismo formato que la CNN; el circuito se reconstruye desde `config_modelo`).
2. Implementar los dos TODO de `backend/app/modelos/qcnn.py`: cargar una vez al arrancar y `probabilidades(ventanas, rr)` llamando a `evaluacion.predecir_probabilidades`.
3. `QCNN_MODO=real` en `.env`. El bloque `resultados.qcnn` del contrato ya tiene la misma forma que el de la CNN y los mismos `indice`/`tiempo_s`.
4. Vigilar el tiempo: el contrato promete menos de 1 s para 5 min de señal. Con simulación, 5 min (~360 latidos) pueden tardar más; si ocurre, se acuerda con el equipo un tiempo mayor o se limita la QCNN a señales cortas (queda reflejado en `estado`/`mensaje`).

## 7. Hitos

| # | Hito | Criterio de aceptación |
| :--- | :--- | :--- |
| 1 | Dependencias y humo | `uv add qiskit qiskit-machine-learning`; un circuito de 8 qubits con `EstimatorQNN` + `TorchConnector` hace un paso de entrenamiento. |
| 2 | QCNN con front-end A | `forward(x, rr)` con la forma correcta; corre con `--rapido`; test de forma y de ensamble. |
| 3 | CV completa de una variante | `results/qcnn/` con OOF, umbral y prueba, siguiendo `ejecutar_cv`. |
| 4 | Ablación | Tabla comparativa (A/B, capas, RR) frente a la CNN. |
| 5 | Backend | `QCNN_MODO=real`; `/api/v1/analisis` devuelve `qcnn.estado == "ok"` con `ejemplo_ecg.csv`. |
| 6 | Reporte | Comparación CNN frente a QCNN con limitaciones (simulación, costo, desviaciones del protocolo). |

## 8. Riesgos

- **Meseta estéril (barren plateaus):** pocos qubits y circuitos poco profundos lo mitigan; se inicializan los pesos con valores pequeños.
- **Cuello de botella en 8 qubits:** la información de la forma de onda debe caber en 8 números; si el F1 queda muy por debajo, subir a 10–12 qubits (el costo de simulación crece de forma exponencial).
- **Compatibilidad de versiones** entre `qiskit`, `qiskit-machine-learning` y `qiskit-aer`: fijar versiones en `uv.lock` y comprobar `uv sync` en limpio.
- **Comparación injusta:** si se submuestrea el entrenamiento o se usan menos semillas, se declara en el reporte.
