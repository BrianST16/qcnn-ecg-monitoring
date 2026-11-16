# Contrato de la API: análisis de ECG con CNN y QCNN

**Proyecto:** Comparación de CNN y QCNN para clasificación de señales ECG (UdeA, 2026-2)
**Versión del contrato:** v1 (borrador para el Ciclo 3)
**Para:** el equipo que construye el backend (FastAPI) y el frontend (React)

> ⚠️ El sistema es académico. Toda respuesta y toda pantalla deben mostrar: *"Resultado experimental con fines académicos. No constituye un diagnóstico médico."*

---

## 1. Qué hace el sistema

1. El usuario sube un archivo con su ECG (una derivación, en mV).
2. React lee el archivo **localmente** y grafica la señal de inmediato, sin esperar al backend.
3. React envía el archivo al backend (`POST /api/v1/analisis`).
4. El backend detecta cada latido, lo recorta y lo clasifica como **NORMAL** o **ANORMAL** con cada modelo: hoy la CNN; la QCNN llega en el Ciclo 3.
5. La respuesta trae un **resumen** y el **resultado de cada latido** con su tiempo en segundos. React usa ese tiempo para marcar los latidos sobre el gráfico.

```
Usuario ──archivo.csv──▶ React ──(1) grafica localmente
                           │
                           └──(2) POST /api/v1/analisis (multipart) ──▶ FastAPI
                                                                          │ valida y remuestrea a 360 Hz
                                                                          │ detecta picos R
                                                                          │ filtra y normaliza
                                                                          │ recorta ventanas de 0.8 s
                                                                          │ CNN (y QCNN)
React ◀──(3) JSON: resumen + latidos[tiempo_s, clase, probabilidad] ◀────┘
   └── marca cada latido sobre el gráfico (verde = normal, rojo = anormal)
```

---

## 2. Archivo de entrada (lo que sube el usuario)

### Formato: CSV de una derivación

Se eligió CSV porque cualquier usuario lo abre en Excel, cualquier equipo lo exporta, y React lo parsea en el navegador con PapaParse para graficar sin pasar por el backend.

| Regla | Valor |
| :--- | :--- |
| Extensión / tipo | `.csv` / `text/csv` |
| Codificación | UTF-8 (con o sin BOM) |
| Separador | coma `,` |
| Decimal | punto `.` (no coma) |
| Encabezado | obligatorio, exactamente `tiempo_s,mlii_mv` |
| `tiempo_s` | segundos desde el inicio, creciente y con paso constante. Se recomiendan 6 decimales. |
| `mlii_mv` | amplitud de la derivación **MLII** (equivalente a la derivación II) en **milivoltios**. Se recomiendan 3 decimales. |
| Frecuencia de muestreo | se **deduce** de `tiempo_s`. Se aceptan de **100 a 1000 Hz**; el modelo trabaja a 360 Hz y el backend remuestrea si hace falta. |
| Duración | de **10 s** a **5 min** |
| Tamaño máximo | **5 MB** (5 min a 360 Hz pesan unos 1.9 MB) |
| Valores faltantes | no se permiten (ni celdas vacías ni `NaN`) |

### Ejemplo (`ejemplo_ecg.csv`, primeras filas)

```csv
tiempo_s,mlii_mv
0.000000,-0.145
0.002778,-0.145
0.005556,-0.150
0.008333,-0.145
0.011111,-0.125
...
29.994444,-0.265
29.997222,-0.275
```

El archivo completo tiene 30 s (10,800 filas, 188 KB). Es un fragmento **real** de la MIT-BIH Arrhythmia Database (registro 233, segundos 230 a 260): un paciente que el modelo **nunca vio** al entrenar, con 41 latidos normales, 10 ventriculares prematuros y 1 de fusión.

### Validaciones (las hace el backend; el frontend puede adelantarlas)

| Validación | Error |
| :--- | :--- |
| El archivo existe, pesa más de 0 y menos de 5 MB | `ARCHIVO_VACIO` / `ARCHIVO_DEMASIADO_GRANDE` |
| Extensión `.csv` y contenido de texto | `TIPO_NO_SOPORTADO` |
| Encabezado exacto `tiempo_s,mlii_mv` | `COLUMNAS_INVALIDAS` |
| Todos los valores numéricos, sin vacíos | `VALORES_INVALIDOS` (indica la fila) |
| `tiempo_s` estrictamente creciente | `TIEMPO_NO_CRECIENTE` |
| Paso de tiempo constante (desviación < 1 %) | `MUESTREO_IRREGULAR` |
| Frecuencia deducida entre 100 y 1000 Hz | `FRECUENCIA_FUERA_DE_RANGO` |
| Duración entre 10 s y 300 s | `DURACION_INSUFICIENTE` / `DURACION_EXCESIVA` |
| Amplitud dentro de ±10 mV (si no, probablemente no está en mV) | `AMPLITUD_FUERA_DE_RANGO` |
| Se detectan al menos 5 latidos | `SIN_LATIDOS_SUFICIENTES` |

Cómo calcular la frecuencia: `fs = (n_filas - 1) / (tiempo_s[último] - tiempo_s[primero])`. No usar la mediana de las diferencias redondeadas: con 6 decimales da 359.97 en vez de 360.

---

## 3. Endpoints

URL base de desarrollo: `http://localhost:8000/api/v1`

| Método | Ruta | Uso |
| :--- | :--- | :--- |
| `POST` | `/analisis` | Sube el archivo y devuelve el análisis (síncrono, menos de 1 s para 5 min de señal) |
| `GET` | `/modelos` | Metadatos y métricas de referencia de cada modelo (para la pantalla de comparación) |
| `GET` | `/salud` | Comprobación de que el servicio y los modelos están cargados |

### 3.1 `POST /api/v1/analisis` (entrada)

`multipart/form-data` con un solo campo obligatorio:

| Campo | Tipo | Obligatorio | Descripción |
| :--- | :--- | :--- | :--- |
| `archivo` | archivo | sí | El CSV del ECG (sección 2) |
| `modelos` | texto | no | Modelos a ejecutar, separados por coma. Por defecto `cnn,qcnn` |

**Petición HTTP (lo que viaja por la red):**

```http
POST /api/v1/analisis HTTP/1.1
Host: localhost:8000
Content-Type: multipart/form-data; boundary=----ecg123

------ecg123
Content-Disposition: form-data; name="archivo"; filename="ejemplo_ecg.csv"
Content-Type: text/csv

tiempo_s,mlii_mv
0.000000,-0.145
0.002778,-0.145
...
------ecg123
Content-Disposition: form-data; name="modelos"

cnn,qcnn
------ecg123--
```

**Con curl:**

```bash
curl -X POST http://localhost:8000/api/v1/analisis \
  -F "archivo=@ejemplo_ecg.csv;type=text/csv" \
  -F "modelos=cnn,qcnn"
```

**Desde React:**

```js
const datos = new FormData();
datos.append("archivo", archivoSeleccionado); // File del <input type="file" accept=".csv">
const respuesta = await fetch(`${API}/analisis`, { method: "POST", body: datos });
// No fijar Content-Type a mano: el navegador agrega el boundary.
const resultado = await respuesta.json();
if (!respuesta.ok) mostrarError(resultado.error);
```

### 3.2 `POST /api/v1/analisis` (respuesta `200 OK`)

Ejemplo **real** para `ejemplo_ecg.csv`: sale de ejecutar el modelo entrenado sobre ese archivo. Está recortado a 4 latidos; el completo está en `ejemplo_respuesta.json`.

```json
{
  "id_analisis": "09502bba-f7c7-47a2-a3ca-d3c60a86288c",
  "fecha": "2026-10-13T21:39:23Z",
  "advertencia": "Resultado experimental con fines académicos. No constituye un diagnóstico médico.",
  "senal": {
    "nombre_archivo": "ejemplo_ecg.csv",
    "derivacion": "MLII",
    "muestras": 10800,
    "duracion_s": 30.0,
    "fs_original_hz": 360.0,
    "fs_modelo_hz": 360,
    "remuestreada": false
  },
  "deteccion": {
    "metodo": "XQRS (wfdb)",
    "latidos_detectados": 52,
    "latidos_evaluados": 51,
    "latidos_descartados_borde": 1,
    "frecuencia_cardiaca_media_lpm": 104.6
  },
  "resultados": {
    "cnn": {
      "estado": "ok",
      "modelo": {
        "id": "cnn1d-base-v1",
        "nombre": "CNN 1-D",
        "parametros": 11265,
        "umbral": 0.615
      },
      "tiempo_inferencia_ms": 11.6,
      "resumen": {
        "latidos_normales": 43,
        "latidos_anormales": 8,
        "porcentaje_anormal": 15.7
      },
      "latidos": [
        { "indice": 4, "tiempo_s": 3.108, "probabilidad_anormal": 0.3365, "clase": "NORMAL",  "rr_previo_s": 0.581 },
        { "indice": 5, "tiempo_s": 3.697, "probabilidad_anormal": 0.6323, "clase": "ANORMAL", "rr_previo_s": 0.589 },
        { "indice": 6, "tiempo_s": 4.133, "probabilidad_anormal": 0.2301, "clase": "NORMAL",  "rr_previo_s": 0.436 },
        { "indice": 7, "tiempo_s": 4.758, "probabilidad_anormal": 0.9873, "clase": "ANORMAL", "rr_previo_s": 0.625 }
      ]
    },
    "qcnn": {
      "estado": "no_disponible",
      "mensaje": "La QCNN se integrará en el Ciclo 3 del proyecto.",
      "modelo": null,
      "tiempo_inferencia_ms": null,
      "resumen": null,
      "latidos": null
    }
  },
  "tiempo_total_ms": 143.2
}
```

#### Referencia de campos

| Campo | Tipo | Descripción |
| :--- | :--- | :--- |
| `id_analisis` | string (UUID) | Identificador del análisis (para logs o historial) |
| `fecha` | string (ISO 8601, UTC) | Momento del análisis |
| `advertencia` | string | Texto legal que el frontend **debe** mostrar |
| `senal.muestras` | int | Filas del CSV |
| `senal.duracion_s` | float | Duración de la señal |
| `senal.fs_original_hz` | float | Frecuencia deducida del archivo |
| `senal.fs_modelo_hz` | int | Siempre 360 |
| `senal.remuestreada` | bool | `true` si el archivo no venía a 360 Hz |
| `deteccion.latidos_detectados` | int | Picos R encontrados por el detector |
| `deteccion.latidos_evaluados` | int | Latidos clasificados: se descartan los que están a menos de 0.4 s de los bordes, porque su ventana queda incompleta |
| `deteccion.frecuencia_cardiaca_media_lpm` | float | 60 / RR medio, en latidos por minuto |
| `resultados.<modelo>.estado` | `"ok"` · `"no_disponible"` · `"error"` | Si `estado != "ok"`, `modelo`, `resumen` y `latidos` son `null` y `mensaje` explica por qué |
| `resultados.<modelo>.modelo.umbral` | float | Probabilidad a partir de la cual un latido es ANORMAL |
| `resultados.<modelo>.tiempo_inferencia_ms` | float | Solo la inferencia del modelo (sirve para comparar CNN y QCNN) |
| `resultados.<modelo>.resumen` | objeto | Conteos y porcentaje de anormales |
| `latidos[].indice` | int | Posición del latido (0, 1, 2, …), igual para todos los modelos |
| `latidos[].tiempo_s` | float | **Dónde marcar el latido en el gráfico** (posición del pico R, en la misma escala que `tiempo_s` del CSV) |
| `latidos[].probabilidad_anormal` | float 0–1 | Salida del modelo |
| `latidos[].clase` | `"NORMAL"` · `"ANORMAL"` | `probabilidad_anormal >= umbral` |
| `latidos[].rr_previo_s` | float o `null` | Tiempo desde el latido anterior (`null` en el primero) |

**Sobre los modelos:** las claves de `resultados` son siempre `cnn` y `qcnn`. Cuando la QCNN exista, su bloque tendrá la misma forma que el de la CNN y el **mismo** `indice`/`tiempo_s` por latido, de modo que el frontend pueda compararlos latido a latido.

### 3.3 Errores

Todos los errores tienen la misma forma (`ejemplo_error.json`):

```json
{
  "error": {
    "codigo": "DURACION_INSUFICIENTE",
    "mensaje": "La señal dura 6.2 s y el mínimo es 10 s.",
    "detalle": { "duracion_s": 6.2, "minimo_s": 10, "maximo_s": 300 }
  }
}
```

| HTTP | Códigos |
| :--- | :--- |
| `400` | `ARCHIVO_VACIO`, `COLUMNAS_INVALIDAS`, `VALORES_INVALIDOS`, `TIEMPO_NO_CRECIENTE`, `MUESTREO_IRREGULAR` |
| `413` | `ARCHIVO_DEMASIADO_GRANDE` |
| `415` | `TIPO_NO_SOPORTADO` |
| `422` | `FRECUENCIA_FUERA_DE_RANGO`, `DURACION_INSUFICIENTE`, `DURACION_EXCESIVA`, `AMPLITUD_FUERA_DE_RANGO`, `SIN_LATIDOS_SUFICIENTES` |
| `500` | `ERROR_INTERNO` (sin detalles internos en `detalle`) |

`mensaje` está pensado para mostrarse tal cual al usuario. `codigo` es para la lógica del frontend.

### 3.4 `GET /api/v1/modelos`

```json
{
  "modelos": [
    {
      "id": "cnn1d-base-v1",
      "clave": "cnn",
      "nombre": "CNN 1-D",
      "disponible": true,
      "parametros": 11265,
      "umbral": 0.615,
      "entrenado_con": "MIT-BIH, DS1 (22 pacientes)",
      "metricas_prueba": { "f1": 0.61, "exactitud": 0.79, "auc_roc": 0.84, "pacientes": 21 }
    },
    {
      "id": null,
      "clave": "qcnn",
      "nombre": "QCNN (simulada)",
      "disponible": false,
      "parametros": null,
      "umbral": null,
      "entrenado_con": null,
      "metricas_prueba": null
    }
  ]
}
```

### 3.5 `GET /api/v1/salud`

```json
{ "estado": "ok", "modelos_cargados": ["cnn"], "version_api": "1.0.0" }
```

---

## 4. Notas para el backend (FastAPI)

**El procesamiento debe ser idéntico al del entrenamiento.** Reutilicen el paquete `qcnn_ecg` del repo; no reimplementen el filtro ni las ventanas.

| Paso | Cómo | Código del repo |
| :--- | :--- | :--- |
| 1. Leer y validar CSV | `pandas.read_csv` más las validaciones de la sección 2 | — |
| 2. Remuestrear a 360 Hz | `scipy.signal.resample_poly` (solo si `fs != 360`) | — |
| 3. Detectar picos R | `wfdb.processing.xqrs_detect(x, fs=360)` y luego `correct_peaks(..., search_radius=18, smooth_window_size=7)` sobre la señal **cruda** remuestreada | — |
| 4. Filtrar y normalizar | pasa-banda de 0.5 a 40 Hz y z-score sobre **toda** la señal subida | `preprocesamiento.preprocesar_senal` |
| 5. Recortar ventanas | 288 muestras (±0.4 s) alrededor de cada pico; se descartan las de los bordes | `preprocesamiento.extraer_ventanas` |
| 6. Clasificar | cargar el modelo **una vez al iniciar la app**, no en cada petición | `entrenar.cargar_modelo("modelos_entrenados/cnn1d_final.pt")` y `evaluacion.predecir_probabilidades` |
| 7. Etiquetar | `clase = "ANORMAL" if p >= umbral else "NORMAL"`; el umbral viene dentro del `.pt` | — |

- **Rendimiento medido** en un portátil i7-8565U (solo CPU) con el ejemplo de 30 s: 52 latidos, inferencia de 12 ms, petición completa de unos 140 ms. Para 5 min se espera menos de 1 s, así que no hace falta procesamiento asíncrono.
- **Concurrencia:** PyTorch usa varios hilos. Fijen `torch.set_num_threads(...)` según el servidor y ejecuten la inferencia en un threadpool (endpoint `def`, no `async def`, o `run_in_threadpool`).
- **CORS:** habilitar el origen del frontend (por ejemplo, `http://localhost:5173` con Vite).
- **Esquemas:** definan modelos Pydantic a partir de la sección 3.2; así OpenAPI (`/docs`) queda documentado solo.
- **Privacidad:** no guarden el archivo en disco después de responder.
- **Verificación cruzada:** al ejecutar su backend con `ejemplo_ecg.csv`, deberían obtener los mismos 52 latidos detectados y 8 anormales que `ejemplo_respuesta.json`. Puede haber diferencias mínimas en las probabilidades (cuarto decimal).

## 5. Notas para el frontend (React)

- **Lectura local:** `Papa.parse(archivo, { header: true, dynamicTyping: true })`. Con las mismas reglas de la sección 2 pueden validar antes de enviar (encabezado, duración, tamaño) y evitar viajes inútiles al backend.
- **Gráfico:** 5 min a 360 Hz son 108,000 puntos; no los dibujen todos a la vez.
  - Muestren una **ventana de 10 s** con desplazamiento o zoom, y una vista general submuestreada (por ejemplo, un punto cada 10).
  - Librerías que manejan bien muchos puntos: **uPlot**, **Plotly** (`scattergl`) o **Recharts** con submuestreo previo.
  - Eje X: `tiempo_s`. Eje Y: `mlii_mv`. La convención clínica dibuja una cuadrícula de 0.2 s × 0.5 mV.
- **Marcar latidos:** por cada elemento de `resultados.cnn.latidos`, un punto o línea vertical en `x = tiempo_s`, verde si es `NORMAL` y rojo si es `ANORMAL`, con un tooltip con `probabilidad_anormal` y `rr_previo_s`.
- **Comparación CNN / QCNN:** dos tarjetas con el `resumen` y el `tiempo_inferencia_ms` de cada modelo. Si `estado === "no_disponible"`, mostrar `mensaje` en la tarjeta de la QCNN. Opcional: resaltar los latidos donde los dos modelos no coinciden (mismo `indice`, distinta `clase`).
- **Advertencia:** mostrar siempre `advertencia`, en un lugar visible.
- **Errores:** mostrar `error.mensaje`; usar `error.codigo` si hace falta lógica (por ejemplo, resaltar el campo del archivo).

---

## 6. Lo que muestra el ejemplo (para no sobrevender el modelo)

Al comparar `ejemplo_respuesta.json` con las anotaciones de los cardiólogos para ese fragmento:

- El detector encontró los **52 latidos** anotados; 51 se evaluaron y 1 quedó en el borde.
- La CNN clasificó bien **46 de 51**:
  - Detectó **7 de 10** latidos ventriculares prematuros, con probabilidades de 0.97 a 0.997.
  - Marcó como anormal **1 latido normal** (3.70 s, probabilidad 0.63).
  - No detectó **3 ventriculares** y **1 de fusión** (probabilidades de 0.23 a 0.53).

Esto es consistente con el desempeño medido del modelo en pacientes nuevos (F1 ≈ 0.61; ver `results/cnn/resumen.md` en el repo). La interfaz debe presentar el resultado como experimental.

## 7. Archivos que acompañan este documento

| Archivo | Contenido |
| :--- | :--- |
| `contrato_api.md` | Este documento |
| `ejemplo_ecg.csv` | Archivo de entrada válido: 30 s reales de MIT-BIH a 360 Hz |
| `ejemplo_respuesta.json` | Respuesta completa (51 latidos) generada con el modelo real a partir de `ejemplo_ecg.csv` |
| `ejemplo_error.json` | Ejemplo de respuesta de error |

**Origen de los datos de ejemplo:** MIT-BIH Arrhythmia Database (Moody & Mark, 2001), PhysioNet, licencia Open Data Commons Attribution v1.0. Si el ejemplo se publica, hay que citar la fuente.
