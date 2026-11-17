# Guía de integración backend ↔ frontend

Para quien construye el frontend (React). Resume cómo hablar con el backend, qué devuelve y qué hay que tener en cuenta. La referencia completa de campos y validaciones es el [contrato v1](../api/contrato_api.md); esta guía no lo reemplaza, dice qué hacer con él y dónde el backend actual se aparta de él.

> **Obligatorio en toda pantalla con resultados:** mostrar el campo `advertencia`: *"Resultado experimental con fines académicos. No constituye un diagnóstico médico."*

## 1. Levantar el backend

```bash
cd backend
cp .env.example .env        # editar si hace falta (ver sección 6)
PYTHONPATH=. uv run --project .. uvicorn app.main:app --port 8000
```

- Documentación interactiva: `http://localhost:8000/docs` (solo si `DOCS_HABILITADOS=true`).
- Comprobación rápida: `GET http://localhost:8000/api/v1/salud` → `{"estado":"ok","modelos_cargados":["cnn","qcnn"],"version_api":"1.0.0"}`.
- `modelos_cargados` dice qué modelos hay: si no aparece `qcnn`, el servidor está con `QCNN_MODO=desactivada` y su bloque llegará como `no_disponible`.
- Los modelos se cargan **una vez al arrancar**; el arranque con la QCNN tarda más porque importa Qiskit.

## 2. Endpoints

URL base: `http://localhost:8000/api/v1`

| Método | Ruta | Para qué | Autenticación | Límite de peticiones |
| :--- | :--- | :--- | :--- | :--- |
| `POST` | `/analisis` | Subir el CSV y recibir el análisis | `X-API-Key` si el servidor la exige | Sí |
| `GET` | `/modelos` | Fichas de los modelos (pantalla de comparación) | `X-API-Key` si el servidor la exige | Sí |
| `GET` | `/salud` | ¿Está vivo y qué modelos tiene? | No | No |

Entrada de `/analisis`: `multipart/form-data` con `archivo` (el CSV) y, opcional, `modelos` (`"cnn"`, `"qcnn"` o `"cnn,qcnn"`; por defecto los dos). Formato del CSV: encabezado exacto `tiempo_s,mlii_mv`, de 10 s a 5 min, 100–1000 Hz, máximo 5 MB (contrato, sección 2). Hay un archivo de ejemplo válido en `docs/api/ejemplo_ecg.csv`.

## 3. Llamada desde React

```js
const API = "http://localhost:8000/api/v1";

async function analizar(archivo, apiKey) {
  const datos = new FormData();
  datos.append("archivo", archivo);          // File de <input type="file" accept=".csv">
  datos.append("modelos", "cnn,qcnn");
  // No fijar Content-Type a mano: el navegador agrega el boundary.
  const r = await fetch(`${API}/analisis`, {
    method: "POST",
    body: datos,
    headers: apiKey ? { "X-API-Key": apiKey } : {},
  });
  const cuerpo = await r.json();
  if (!r.ok) throw cuerpo.error;            // { codigo, mensaje, detalle }
  return cuerpo;
}
```

## 4. Errores

Todos tienen la misma forma: `{"error": {"codigo", "mensaje", "detalle"}}`. Mostrar `mensaje` tal cual al usuario; usar `codigo` para la lógica.

| HTTP | Códigos | Qué hacer en la interfaz |
| :--- | :--- | :--- |
| `400` | `ARCHIVO_VACIO`, `COLUMNAS_INVALIDAS`, `VALORES_INVALIDOS`, `TIEMPO_NO_CRECIENTE`, `MUESTREO_IRREGULAR` | Error del archivo: mostrar el mensaje junto al selector. |
| `401` | `NO_AUTORIZADO` | Falta o es inválida la `X-API-Key`. |
| `413` / `415` | `ARCHIVO_DEMASIADO_GRANDE` / `TIPO_NO_SOPORTADO` | Validar antes de enviar (5 MB, `.csv`). |
| `422` | `FRECUENCIA_FUERA_DE_RANGO`, `DURACION_INSUFICIENTE`, `DURACION_EXCESIVA`, `AMPLITUD_FUERA_DE_RANGO`, `SIN_LATIDOS_SUFICIENTES`, `SOLICITUD_INVALIDA` | Error del archivo o del campo `modelos`. |
| `429` | `DEMASIADAS_PETICIONES` | Esperar un poco antes de reintentar (por defecto 30 por minuto por IP). |
| `503` | `SERVICIO_OCUPADO` | Hay 2 análisis en curso (por defecto); no hay cola. Reintentar en unos segundos. |
| `500` | `ERROR_INTERNO` | Mensaje genérico; no hay detalle interno. |

Los códigos `NO_AUTORIZADO`, `DEMASIADAS_PETICIONES`, `SERVICIO_OCUPADO` y `SOLICITUD_INVALIDA` no están en la tabla del contrato v1: los añadió el backend.

## 5. Interpretar la respuesta de `/analisis`

- **Un bloque por modelo:** `resultados.cnn` y `resultados.qcnn`, siempre los dos. Mirar `estado` antes de leer el resto: si no es `"ok"`, `modelo`, `resumen` y `latidos` son `null` y `mensaje` explica por qué.
- **Marcar latidos en el gráfico:** `x = latidos[].tiempo_s` (misma escala que `tiempo_s` del CSV); verde si `clase == "NORMAL"`, rojo si `"ANORMAL"`. Tooltip con `probabilidad_anormal` y `rr_previo_s` (`null` en el primero).
- **Comparar CNN y QCNN latido a latido:** ambos bloques traen el mismo `indice` y `tiempo_s` por latido. Se pueden resaltar los que difieren en `clase`.
- **Tarjetas de comparación:** `resumen` y `tiempo_inferencia_ms` de cada modelo.
- **Redondear al mostrar:** el backend devuelve decimales sin redondear (por ejemplo, `umbral` 0.6149861812591553). Mostrar 2–4 decimales.
- **No todos los latidos se evalúan:** `deteccion.latidos_evaluados` puede ser menor que `latidos_detectados` (los de los bordes se descartan). Hay que usar `latidos_evaluados` para porcentajes.

## 6. Configuración del servidor (`backend/.env`)

| Variable | Valor por defecto | Para qué |
| :--- | :--- | :--- |
| `ORIGENES_PERMITIDOS` | `http://localhost:5173` | Orígenes CORS, separados por coma. **Si el frontend corre en otro puerto o dominio, hay que añadirlo.** |
| `HOSTS_PERMITIDOS` | `localhost,127.0.0.1` | Hosts aceptados. Otro host da `400 Invalid host header`. |
| `API_KEY` | sin definir | Si se define (mínimo 20 caracteres), `/analisis` y `/modelos` exigen `X-API-Key`. |
| `MODELO_SIMULADO` | `false` | `true` usa una CNN falsa (para probar sin PyTorch); sus resultados **no son reales**. |
| `QCNN_MODO` | `desactivada` (en el código) | `real` carga la QCNN; `maqueta` una falsa; `desactivada` devuelve `no_disponible`. `.env.example` ya trae `real`. |
| `MAX_ANALISIS_SIMULTANEOS` | `2` | Tope de análisis a la vez (si se pasa: `503`). |
| `LIMITE_PETICIONES_POR_MINUTO` | `30` | Por IP (si se pasa: `429`). |
| `DOCS_HABILITADOS` | `true` | Publica `/docs`. Para producción conviene `false`. |

## 7. Cosas a tener en cuenta

**Seguridad**
- **Una `X-API-Key` en una SPA queda visible** para cualquiera que abra las herramientas del navegador. Sirve para tener un candado básico, no para proteger datos sensibles. Si se publica la aplicación, el frontend debería hablar con un proxy que guarde la clave.
- El backend no guarda el archivo ni datos del paciente; los registros solo llevan cifras del análisis.
- CORS solo permite `GET`, `POST` y la cabecera `X-API-Key`.

**Resultados de los modelos (para no sobrevender)**
- Son experimentales y se evalúan con pacientes que los modelos no vieron, pero el rendimiento varía mucho entre pacientes y entre semillas de entrenamiento (F1 de la CNN entre 0.5 y 0.65).
- **La QCNN tiene un umbral problemático.** Su umbral (≈ 0.28) se calculó con modelos individuales y se aplica a un ensamble de 5 que promedia sus probabilidades. En el ejemplo `ejemplo_ecg.csv` marca 46 de 51 latidos como anormales, contra 8 de la CNN, mientras que las anotaciones indican 10 ventriculares y 1 de fusión. **No presentar el resultado de la QCNN como fiable hasta que se recalibre el umbral** (hay un recálculo en curso con validación cruzada anidada). El frontend no necesita cambios cuando eso ocurra: solo cambiarán las probabilidades y el umbral.
- `parametros` del bloque `modelo` es la **suma de todos los modelos del ensamble** (CNN: 11 265; QCNN: 4 640, es decir, 5 × 928).
- **Tiempos:** el ejemplo de 30 s tarda unos 0.2 s en total, con unos 130 ms de QCNN. Para 5 min la QCNN tardaría del orden de 1 s (estimado, sin medir). La QCNN es un circuito cuántico **simulado**: no hay ventaja de velocidad.
- La QCNN se llama `"QCNN (simulada)"` en todo el sistema: no ocultar la palabra *simulada*.

**Diferencias con el contrato v1 que el frontend ya encontrará**
- `GET /modelos` devuelve `entrenado_con` y `metricas_prueba` en `null` aunque el modelo esté cargado (el contrato los muestra con valores). No dependan de ellos hasta que el backend los complete.
- La frase del contrato *"La QCNN se integrará en el Ciclo 3"* solo aparece cuando la QCNN está desactivada.
- `docs/api/ejemplo_respuesta.json` fue generado con la CNN y la QCNN `no_disponible`; con `QCNN_MODO=real` el bloque `qcnn` viene lleno.

**Rendimiento y diseño**
- Un análisis es **síncrono** (menos de 1 s para el ejemplo); no hace falta *polling*. Mostrar un indicador de carga y desactivar el botón mientras dura la petición.
- 5 min de señal son 108 000 puntos: graficar una ventana de 10 s con zoom y una vista general submuestreada (uPlot o Plotly `scattergl`).
- React puede graficar el CSV **antes** de que responda el backend, y validar encabezado, duración y tamaño localmente para evitar viajes inútiles.

## 8. Probar sin frontend

```bash
curl -X POST http://localhost:8000/api/v1/analisis \
  -F "archivo=@docs/api/ejemplo_ecg.csv;type=text/csv" \
  -F "modelos=cnn,qcnn"
```

Con `API_KEY` definida, añadir `-H "X-API-Key: <clave>"`. Resultado esperado con `ejemplo_ecg.csv`: 52 latidos detectados, 51 evaluados.

Pruebas del backend: `cd backend && PYTHONPATH=. uv run --project .. pytest tests` (82 tests).
