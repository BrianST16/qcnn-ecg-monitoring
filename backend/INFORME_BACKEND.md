# Informe de avance: backend FastAPI (CNN y QCNN)

Nov 11, 2026 · @Brayan

## Resumen

Este informe describe el backend (FastAPI) del prototipo que clasifica latidos de ECG con una CNN y una QCNN. Se construyó paso a paso siguiendo el contrato de la API (`contrato_api.md`). El backend está **funcionalmente completo para la CNN**: valida el CSV, detecta los latidos, los clasifica y responde con el formato del contrato. Hoy la clasificación la hace un **modelo simulado** con la misma interfaz que el real; el preprocesamiento es código real de `qcnn_ecg`.

**Estado.** Los pasos 1 a 10C están completados y verificados: 68 pruebas unitarias y pruebas HTTP manuales de cada capa. Las 14 pruebas de seguridad HTTP (paso 11A) están escritas y por ejecutar. El despliegue está planeado y documentado (paso 11B), pero **no se ha hecho**. Los dos modelos, CNN y QCNN, tienen su hueco listo en el backend: conectar cada uno está explicado en la sección «Qué subir, conectar y modificar». El código se entrega en la rama `backend` para revisión, sin pull request a `main`. No se guarda nada de los análisis, y los secretos viven en un `.env` que no se sube.

## Qué hace el backend

Recibe el ECG de una derivación en un CSV, lo valida, detecta cada latido, lo clasifica como NORMAL o ANORMAL y devuelve un JSON con el resumen y el resultado de cada latido, con el formato exacto del contrato.

```text
CSV (tiempo_s, mlii_mv)
  -> validar: tamaño (máx. 5 MB), columnas, tiempo, frecuencia (100 a 1000 Hz), duración (10 s a 5 min), amplitud
  -> remuestrear a 360 Hz si hace falta
  -> detectar picos R (XQRS, wfdb)
  -> filtrar (0.5 a 40 Hz) y normalizar
  -> recortar una ventana de 288 muestras por latido
  -> CNN (y QCNN) clasifican cada ventana
  -> JSON: resumen + latidos con tiempo, probabilidad y clase
```

| Método y ruta | Qué hace | Acceso |
| --- | --- | --- |
| `GET /api/v1/salud` | Dice si el servicio está vivo y qué modelos hay cargados | Público |
| `GET /api/v1/modelos` | Ficha de cada modelo (id, parámetros, umbral, disponibilidad) | Clave `X-API-Key` y límite por IP |
| `POST /api/v1/analisis` | Recibe el CSV y devuelve el análisis | Clave `X-API-Key` y límite por IP |

**Qué es real y qué es simulado.** La lectura y validación del CSV, el remuestreo, la detección de picos R, el filtrado y las ventanas son **código real**, con las funciones del paquete `qcnn_ecg` que se usaron para entrenar. Hoy lo que se simula es la **clasificación**: la CNN usa un modelo simulado con la misma interfaz que el real, y la QCNN está desactivada y responde `no_disponible`. Conectar los modelos reales está explicado en la sección «Qué subir, conectar y modificar».

## Mapa de archivos

Todo está en la carpeta `backend/`.

| Archivo | Responsabilidad |
| --- | --- |
| `app/main.py` | Crea la app, monta las capas de red y define las tres rutas |
| `app/config.py` | Lee la configuración del `.env` y falla al arrancar si algo es inválido |
| `app/errores.py` | Excepción propia, códigos de error y formato de error del contrato |
| `app/esquemas.py` | Esquemas Pydantic de las respuestas |
| `app/seguridad.py` | Dependencia que exige la API key |
| `app/limite_cuerpo.py` | Middleware que corta con 413 los cuerpos demasiado grandes |
| `app/limitador.py` | Límite de peticiones por IP (429) |
| `app/cupo.py` | Tope de análisis simultáneos (503) |
| `app/archivos.py` | Lectura del archivo subido: máximo 5 MB y texto UTF-8 |
| `app/csv_ecg.py` | Validación del CSV y lectura de la señal |
| `app/procesamiento.py` | Remuestreo, picos R, filtrado y ventanas (usa `qcnn_ecg`) |
| `app/analisis.py` | Ejecuta los modelos y arma la respuesta del contrato |
| `app/modelos/base.py` | Interfaz `Clasificador` que debe cumplir todo modelo |
| `app/modelos/simulado.py` | `CNNSimulada` y `QCNNMaqueta`, para probar sin modelos reales |
| `app/modelos/cnn.py` | Adaptador de la CNN real, que reutiliza `qcnn_ecg` |
| `app/modelos/qcnn.py` | Hueco de la QCNN real, con dos TODO por rellenar |
| `app/modelos/registro.py` | Decide qué modelos se cargan, según la configuración |
| `tests/` | 68 pruebas unitarias, más 14 de seguridad HTTP escritas y por ejecutar |
| `.env.example` | Plantilla de configuración, sin secretos |

## Cómo ejecutarlo y probarlo

Desde PowerShell. Cada persona crea su propio `.env`: el real no está en el repositorio.

```powershell
uv sync                                   # en la raíz del repo: instala las dependencias
cd backend
copy .env.example .env                    # luego completar API_KEY (mínimo 20 caracteres)
uv run python -c "import secrets; print(secrets.token_urlsafe(32))"   # genera una clave
uv run uvicorn app.main:app --reload      # http://127.0.0.1:8000 (con /docs)
uv run python -m pytest tests -q          # corre las pruebas
```

Para probar un análisis con el archivo de ejemplo `Archivos/ejemplo_ecg.csv` (30 s reales de MIT-BIH):

```powershell
curl.exe -sS -H "X-API-Key: <la clave del .env>" -F "archivo=@..\Archivos\ejemplo_ecg.csv;type=text/csv" http://127.0.0.1:8000/api/v1/analisis
```

Con el modelo simulado, la respuesta trae 52 latidos detectados, 51 evaluados y 104.6 lpm, que coinciden con el ejemplo del contrato. Los latidos anormales difieren, porque la clasificación es simulada.

## Para quien revisa la rama

- **Rama:** el código está en la rama `backend`. No se ha hecho pull request a `main`: queda para revisión. Este informe también está en el archivo backend/INFORME\_BACKEND.md de la rama.
- **Por dónde empezar a leer:** `app/main.py` (la vista general), luego `app/analisis.py` y `app/procesamiento.py` (el núcleo), `app/modelos/` (cómo se conectan los modelos) y por último las capas de seguridad (`seguridad.py`, `limite_cuerpo.py`, `limitador.py`, `cupo.py`).
- **Lo que no está en el repo:** el `.env` con los secretos. La API key no aparece en ningún archivo ni en este informe.
- **Archivos compartidos con el equipo de modelos:** `pyproject.toml` y `uv.lock` cambiaron al añadir `fastapi`, `uvicorn`, `python-multipart` y `pydantic-settings` (y `httpx` como dependencia de desarrollo para las pruebas 11A). Conviene comprobar que `uv sync` siga funcionando.
- **Lo que el backend no modifica:** el paquete `src/qcnn_ecg` y la carpeta `modelos_entrenados/`. Solo los importa y los lee.
- **Decisiones que necesitan respuesta:** están reunidas en la sección «Puntos abiertos», antes del estado final.

## Decisiones de alcance y seguridad

El backend se despliega como servidor público temporal para una demo, con API key compartida y límite de peticiones por IP. No se guarda nada de los análisis.

| Decisión | Elección |
| --- | --- |
| Alcance | Solo backend (FastAPI). La CNN queda lista para conectar el modelo final y la QCNN tiene su hueco preparado. Los modelos y el frontend los hace otra persona. |
| Despliegue | Servidor público temporal para la demo. El plan está en el paso 11B y aún no se ha desplegado. |
| Autenticación | API key compartida en el header `X-API-Key` |
| Rate limiting | Sí, por IP |
| Datos guardados | Nada: ni el archivo ni historial |
| Modelo CNN | Simulado por ahora; el real se conecta cambiando la configuración |
| QCNN | Desactivada por defecto: devuelve `"estado": "no_disponible"`, como en el contrato. Tiene una maqueta solo para pruebas. |

### Capas de seguridad

| Capa | Se implementa en | Estado |
| --- | --- | --- |
| Secretos fuera del repo (`.env`) | Paso 2 | Hecha |
| Documentación `/docs` apagable | Paso 2 | Hecha y probada |
| Errores genéricos (`ERROR_INTERNO`) | Paso 5 | Hecha y probada |
| API key con `X-API-Key` | Paso 6 (la clave ya se configura en el paso 2) | Hecha: protege /modelos y /analisis |
| Límite de subida de 5 MB y tipo de archivo | Paso 7 | Hecha y probada: tipo validado, lectura limitada a 5 MB y corte con 413 antes de procesar el cuerpo |
| Sin persistencia ni logs con datos | Pasos 8 y 9 | Hecha y probada: todo en memoria, y el log de cada análisis solo lleva id, latidos y milisegundos |
| Tope de análisis simultáneos | Paso 9 | Hecha y probada: con tope 2, de 5 peticiones simultáneas 2 reciben 200 y 3 reciben 503 |
| CORS estricto, hosts permitidos, rate limiting | Paso 10 | Hecha y probada: CORS, hosts permitidos y 30 peticiones por minuto por IP (429) |
| HTTPS | Paso 11B, lo aporta el túnel de Cloudflare | Plan documentado (paso 11B); no desplegado |

La API key no es una protección fuerte: React la expone en el navegador. La defensa real contra abuso son el rate limiting y los límites de tamaño.

### Cambio al contrato que hay que comunicar

Se proponen cuatro códigos de error nuevos, con el formato de `ejemplo_error.json`. Quien haga el frontend debe mostrarlos y enviar `X-API-Key` en cada petición.

| HTTP | Código | Cuándo |
| --- | --- | --- |
| 401 | `NO_AUTORIZADO` | Falta la clave o es incorrecta |
| 429 | `DEMASIADAS_PETICIONES` | Se superó el límite por IP |
| 503 | `SERVICIO_OCUPADO` | Se alcanzó el tope de análisis simultáneos |
| 404, 405 o 422 | SOLICITUD\_INVALIDA | Petición mal formada, ruta inexistente o método incorrecto |

## Paso 1: entorno y endpoint de salud (completado)

El servidor arranca en Windows desde VS Code y responde en `/api/v1/salud` y `/docs`.

- Se instaló `uv` y se corrió `uv sync`, que creó `.venv` con las dependencias del proyecto.
- Se añadieron `fastapi`, `uvicorn[standard]` y `python-multipart` con `uv add`. Esto modificó `pyproject.toml` y `uv.lock`, que comparte el equipo de modelos.
- Se creó `backend/app/__init__.py` (vacío) y `backend/app/main.py` con `GET /api/v1/salud`.
- El endpoint es `def` y no `async def`, para que la inferencia de PyTorch no bloquee el servidor.
- Se arranca desde `backend/` con `uv run uvicorn app.main:app --reload`.

**Verificación.** `/docs` muestra «API de análisis de ECG (CNN y QCNN)» versión 1.0.0 (OAS 3.1). `/api/v1/salud` devuelve `{"estado":"ok","modelos_cargados":[],"version_api":"1.0.0"}`.

**Recomendación.** Trabajar en una rama `backend`, porque `pyproject.toml` y `uv.lock` son compartidos.

## Paso 2: configuración y secretos (completado)

La API key y los parámetros de entorno viven en `backend/.env`, que no se sube a GitHub, y se leen con `pydantic-settings`. Si falta un valor o es inválido, el servidor no arranca.

| Archivo | Qué contiene |
| --- | --- |
| `backend/app/config.py` | Clase `Ajustes` con `api_key` (mínimo 20 caracteres), `origenes_permitidos` (texto separado por comas) y `docs_habilitados`. `get_ajustes()` usa `lru_cache` para leerla una sola vez. |
| `backend/.env` | Valores reales. Lo excluye el `.gitignore` del repo. |
| `backend/.env.example` | Plantilla sin secretos. Esta sí se sube. |
| `backend/app/main.py` | Apaga `/docs`, `/redoc` y `/openapi.json` cuando `DOCS_HABILITADOS=false`. |

Se añadió la dependencia `pydantic-settings` con `uv add`. La ruta del `.env` se calcula desde `__file__`, así que se encuentra sin importar desde dónde se lance el servidor.

### Pruebas ejecutadas

Las hizo el usuario en su equipo y todas dieron el resultado esperado.

| Prueba | Resultado esperado | Resultado |
| --- | --- | --- |
| Arranque normal | `/salud` y `/docs` responden | Correcto |
| `DOCS_HABILITADOS=false` | `/docs` da «Not Found» y `/salud` sigue activo | Correcto |
| `API_KEY=corta` | El servidor no arranca y pide al menos 20 caracteres | Correcto |
| Sin archivo `.env` | El servidor no arranca: falta `api_key` | Correcto |

## Paso 3: servicio de modelos con CNN simulada (completado)

La app carga los modelos una sola vez al iniciar, con `lifespan`, detrás de una interfaz común. Hoy la cumple una CNN simulada; la CNN real la cumplirá después sin cambiar los endpoints.

| Archivo | Función |
| --- | --- |
| `app/modelos/base.py` | Clase abstracta `Clasificador`: metadatos (`clave`, `id`, `nombre`, `parametros`, `umbral`) y el método `probabilidades(ventanas)`, que recibe `(n, 288)` y devuelve `(n,)` con valores entre 0 y 1 |
| `app/modelos/simulado.py` | `CNNSimulada`: probabilidades aleatorias con semilla fija, cerca de 15 % anormales. No lee la señal. |
| `app/modelos/registro.py` | `cargar_modelos(ajustes)` devuelve `{"cnn": ...}`. Con `MODELO_SIMULADO=false` lanza `NotImplementedError` hasta conectar el modelo real. |
| `app/config.py` | Nuevo campo `modelo_simulado`, por defecto `true` |
| `app/main.py` | `lifespan` guarda los modelos en `app.state.modelos`; `/salud` los lista |

Mientras el modelo sea simulado, su `id` es `cnn1d-simulada` y su `nombre` termina en «(SIMULADA)», para que nadie confunda esas clasificaciones con las reales. El servidor también escribe una advertencia al arrancar.

**Pruebas confirmadas.** Con el modelo simulado, `/salud` debe mostrar `["cnn"]` en `modelos_cargados`. Con `MODELO_SIMULADO=false`, el servidor debe negarse a arrancar con un mensaje claro. La CNN simulada, llamada con 10 ventanas de ceros, debe devolver 10 valores entre 0 y 1.

## Paso 4: esquemas Pydantic y `GET /modelos` (completado)

Los esquemas de `app/esquemas.py` describen las respuestas del contrato, y `GET /api/v1/modelos` ya devuelve la ficha de cada modelo.

| Archivo | Qué contiene |
| --- | --- |
| `app/esquemas.py` | 12 clases. `RespuestaSalud`, `RespuestaModelos` y `FichaModelo` se usan ya; `RespuestaAnalisis` y sus clases anidadas (sección 3.2 del contrato) se conectan en el paso 9. |
| `app/modelos/base.py` | Dos atributos opcionales nuevos en `Clasificador`: `entrenado_con` y `metricas_prueba` |
| `app/main.py` | `response_model` en `/salud` y nuevo endpoint `GET /api/v1/modelos` |

El modelo simulado deja `entrenado_con` y `metricas_prueba` en `null` a propósito: no se inventan métricas para un modelo falso. En el bloque `ok` de un resultado, `mensaje` sale como `null`; el contrato solo lo usa para explicar estados distintos de `ok`.

### Pruebas confirmadas

| Prueba | Resultado esperado | Resultado |
| --- | --- | --- |
| `/api/v1/salud` | Igual que antes | Correcto |
| `/api/v1/modelos` | CNN simulada disponible y QCNN no disponible | Correcto |
| `/docs`, sección Schemas | Aparecen los esquemas de los endpoints | Correcto |
| `RespuestaAnalisis` contra `ejemplo_respuesta.json` | 51 latidos y 8 anormales | Correcto |

## Paso 5: manejadores de error globales (completado)

Todos los errores salen con la forma `{"error": {"codigo", "mensaje", "detalle"}}` del contrato (sección 3.3) y sin detalles internos.

| Archivo | Qué contiene |
| --- | --- |
| `app/errores.py` | `ErrorAPI` (excepción propia), la tabla `ESTADOS` que relaciona cada código con su estado HTTP y `registrar_manejadores` |
| `app/esquemas.py` | `CuerpoError` y `RespuestaError`, que fijan la forma del error |
| `app/main.py` | Registra los manejadores con `trazas=ajustes.docs_habilitados` |

| Manejador | Qué atrapa | Qué responde |
| --- | --- | --- |
| `ErrorAPI` | Errores esperados del contrato | El código y el estado de la tabla `ESTADOS` |
| `RequestValidationError` | Petición mal formada | 400 `ARCHIVO_VACIO` si falta `archivo`; en otro caso 422 `SOLICITUD_INVALIDA` |
| `StarletteHTTPException` | Rutas inexistentes y métodos incorrectos | 404 o 405 con `SOLICITUD_INVALIDA` |
| `Exception` | Cualquier fallo inesperado | 500 `ERROR_INTERNO` genérico. Con `/docs` apagado, el log no incluye el mensaje de la excepción, porque podría traer un fragmento del CSV. |

El código `SOLICITUD_INVALIDA` no existe en el contrato: se suma a `NO_AUTORIZADO`, `DEMASIADAS_PETICIONES` y `SERVICIO_OCUPADO` como cuarto código nuevo que el frontend debe conocer. La ruta temporal de prueba se eliminó al terminar.

### Pruebas confirmadas

| Prueba | Resultado esperado | Resultado |
| --- | --- | --- |
| Error esperado (`DURACION_INSUFICIENTE`) | Idéntico a `ejemplo_error.json`, con estado 422 | Correcto |
| Error inesperado | `ERROR_INTERNO` genérico, sin el texto interno; traceback solo en la consola | Correcto |
| Ruta inexistente | 404 `SOLICITUD_INVALIDA` | Correcto |
| Método incorrecto (`POST` a `/salud`) | 405 `SOLICITUD_INVALIDA` | Correcto |
| Limpieza de la ruta temporal | La ruta responde 404 | Correcto |

## Paso 6: API key por header (completado)

`/modelos` exige el header `X-API-Key` con la clave del `.env`; `/salud` sigue pública. Si la clave falta o es incorrecta, la respuesta es 401 `NO_AUTORIZADO`, con el mismo mensaje en ambos casos para no indicar cuál falló.

| Archivo | Qué contiene |
| --- | --- |
| `app/seguridad.py` | `exigir_api_key`: dependencia que lee `X-API-Key` y la compara con `secrets.compare_digest`. Usa `auto_error=False` para lanzar `NO_AUTORIZADO` en lugar del 403 por defecto de FastAPI. |
| `app/main.py` | `dependencies=[Depends(exigir_api_key)]` en `/modelos`. `/analisis` se protege igual en el paso 9. |

**Qué debe saber el equipo.** El frontend tiene que enviar `X-API-Key` en cada petición a `/modelos` y `/analisis`. La clave real se entrega por un canal privado y nunca va en el repo. En React la clave queda visible en el navegador: frena el uso casual, pero no es una protección fuerte.

**Al conectar el modelo real.** No cambia nada de la seguridad. Solo hay que usar una `API_KEY` nueva en el servidor.

### Pruebas confirmadas

| Prueba | Resultado esperado | Resultado |
| --- | --- | --- |
| `/salud` sin clave | 200 | Correcto |
| `/modelos` sin clave | 401 `NO_AUTORIZADO` | Correcto |
| `/modelos` con clave incorrecta | 401, mismo mensaje | Correcto |
| `/modelos` con clave correcta (desde `/docs` con Authorize) | 200 | Correcto |

## Paso 7: subida y validación del CSV (completado)

La API ya recibe el archivo, lo lee y lo valida con las reglas de la sección 2 del contrato; si algo falla, responde con el código de error correspondiente. El archivo nunca se guarda: su texto vive en memoria solo mientras dura la petición.

| Archivo | Qué hace |
| --- | --- |
| `app/archivos.py` (7A) | Recibe el `UploadFile` y lee como máximo 5 MB + 1 byte. Lanza `ARCHIVO_VACIO`, `ARCHIVO_DEMASIADO_GRANDE` o `TIPO_NO_SOPORTADO` (extensión distinta de `.csv` o texto que no es UTF-8). Devuelve el texto decodificado y acepta BOM. |
| `app/csv_ecg.py` (7B) | `leer_senal(texto)` aplica las reglas del contrato en el orden de su tabla y devuelve la señal con su frecuencia y duración deducidas. |
| `tests/test_csv_ecg.py` | 15 pruebas unitarias con CSV sintéticos. |
| `app/main.py` | Ruta temporal `POST /api/v1/_prueba_archivo`, protegida con la API key, para probar los dos módulos. |

Reglas que aplica `leer_senal`, en orden:

| Orden | Regla | Código de error |
| --- | --- | --- |
| 1 | Encabezado exactamente `tiempo_s,mlii_mv` | `COLUMNAS_INVALIDAS` |
| 2 | Dos columnas por fila, sin celdas vacías ni valores no numéricos (indica la fila) | `VALORES_INVALIDOS` |
| 3 | Al menos 2 filas de datos | `DURACION_INSUFICIENTE` |
| 4 | Tiempo estrictamente creciente | `TIEMPO_NO_CRECIENTE` |
| 5 | Paso de tiempo constante (desviación máxima < 1 %) | `MUESTREO_IRREGULAR` |
| 6 | Frecuencia deducida entre 100 y 1000 Hz | `FRECUENCIA_FUERA_DE_RANGO` |
| 7 | Duración entre 10 y 300 s | `DURACION_INSUFICIENTE` o `DURACION_EXCESIVA` |
| 8 | Amplitud dentro de ±10 mV | `AMPLITUD_FUERA_DE_RANGO` |

**Interpretaciones por confirmar con quien escribió el contrato.** El contrato pide una desviación del paso «< 1 %» sin precisar la medida: se usa la desviación máxima respecto al paso medio, que detecta una muestra faltante. La duración se calcula como número de muestras ÷ frecuencia, redondeada a centésimas, para que un archivo de exactamente 10 s no se rechace por redondeo de coma flotante.

**Ruta temporal.** `POST /api/v1/_prueba_archivo` solo servía para probar. Se reemplazó por `POST /api/v1/analisis` en el paso 9B y ya fue borrada.

**Para el paso 8.** La frecuencia deducida de `ejemplo_ecg.csv` es 360,00001 Hz y no exactamente 360, así que la decisión de remuestrear necesita una tolerancia (por ejemplo, |fs − 360| < 0,5 Hz).

**Al conectar el modelo real.** Este paso no se modifica: la validación del archivo no depende del modelo. El modelo entra en el paso 9, a través de la interfaz `Clasificador` (método `probabilidades`); lo que hay que subir, conectar y modificar sigue siendo lo de la sección «Qué subir, conectar y modificar».

### Pruebas confirmadas

| Prueba | Resultado esperado | Resultado |
| --- | --- | --- |
| `pytest tests -q` | 15 passed | Correcto |
| `ejemplo_ecg.csv` con la clave correcta | 200, 10800 muestras, 360 Hz, 30 s | Correcto |
| CSV con encabezado incorrecto | 400 `COLUMNAS_INVALIDAS` | Correcto |

## Paso 8: remuestreo, picos R y ventanas (completado)

`procesar_senal` toma la señal ya validada y aplica los pasos 2 a 5 de la sección 4 del contrato: remuestrea a 360 Hz, detecta los picos R, filtra y normaliza toda la señal y recorta una ventana de 288 muestras por latido. Todo ocurre en memoria; no se escribe nada en disco.

| Archivo | Qué hace |
| --- | --- |
| `app/procesamiento.py` | `remuestrear_a_modelo`, `detectar_picos_r` y `procesar_senal`, que devuelve las ventanas, los tiempos y los RR de cada latido evaluado. |
| `tests/test_procesamiento.py` | 13 pruebas con un detector falso, para no depender de `wfdb`. |
| `app/main.py` | La ruta temporal `_prueba_archivo` ahora devuelve el resumen del procesamiento (se quitó el campo `modelos`, que vuelve en el paso 9). |

| Paso del contrato | Cómo se hace | Dónde |
| --- | --- | --- |
| Remuestrear a 360 Hz | `resample_poly` con razón de enteros, solo si la frecuencia deducida se aparta de 360 Hz en 0,5 Hz o más | `scipy` |
| Detectar picos R | `xqrs_detect` y luego `correct_peaks` (radio 18, ventana 7) sobre la señal cruda remuestreada | `wfdb` |
| Filtrar y normalizar | Pasa-banda de 0,5 a 40 Hz y z-score sobre toda la señal | `qcnn_ecg.preprocesamiento.preprocesar_senal` |
| Recortar ventanas | 288 muestras (±0,4 s) por pico; se descartan los de los bordes | `qcnn_ecg.preprocesamiento.extraer_ventanas` |

**Decisiones.** El detector se pasa como parámetro de `procesar_senal`, así las pruebas usan picos conocidos y `wfdb` se importa solo al detectar. El RR de cada latido se calcula con todos los picos detectados, incluidos los descartados por el borde, igual que en `ejemplo_respuesta.json`. El tiempo de cada latido suma `tiempo_s[0]` del CSV, para quedar en la misma escala. Con menos de 5 latidos se lanza `SIN_LATIDOS_SUFICIENTES`. La tolerancia de 0,5 Hz resuelve el aviso del paso 7: la frecuencia deducida de `ejemplo_ecg.csv` es 360,00001 Hz y no se remuestrea.

**Al conectar el modelo real.** Este paso no se modifica: el procesamiento es el mismo para la CNN simulada y la real. Solo exige que el entorno del servidor tenga `qcnn_ecg`, `wfdb` y `scipy`, que ya están en el `pyproject.toml` del repo.

### Pruebas confirmadas

| Prueba | Resultado esperado | Resultado |
| --- | --- | --- |
| El entorno importa `wfdb` y `qcnn_ecg` | Sin errores | Correcto |
| `pytest tests -q` | 28 passed | Correcto |
| `ejemplo_ecg.csv` con la clave correcta | 52 detectados, 51 evaluados, 1 descartado, FC 104,6, ventanas (51, 288) | Correcto |
| Señal plana de 10 s | 422 `SIN_LATIDOS_SUFICIENTES` | Correcto |

## Paso 9: ensamblar `POST /analisis` (completado)

El paso 9 va en dos partes. La 9A armó y probó, sin servidor, la lógica que convierte el procesamiento en el JSON del contrato (sección 3.2). La 9B conecta el endpoint, el tope de análisis simultáneos y el log limpio, y borra la ruta temporal.

| Archivo | Qué hace |
| --- | --- |
| `app/analisis.py` | `interpretar_modelos` (campo `modelos`), `ejecutar_modelo` (clase por latido, resumen y tiempo de inferencia), `armar_resultados` (siempre los bloques `cnn` y `qcnn`) y `armar_respuesta` (respuesta 200 completa). |
| `tests/test_analisis.py` | 13 pruebas con modelos falsos: umbral exacto, modelo que falla, forma incorrecta y forma del JSON. |

**Interpretaciones por confirmar con quien escribió el contrato.** Si piden solo `cnn`, el bloque `qcnn` sigue presente como `no_disponible` con el mensaje «No se solicitó este modelo en la petición», porque el contrato dice que las claves son siempre las dos. Si `modelos` trae algo desconocido o queda vacío, se responde `SOLICITUD_INVALIDA` (422). Si un modelo falla, el HTTP es 200 y ese bloque lleva `estado: "error"`, que el contrato prevé; en el log queda solo el tipo de error, nunca su mensaje.

**Hallazgo de las pruebas.** `SOLICITUD_INVALIDA` no estaba en el diccionario `ESTADOS` de `app/errores.py`, porque hasta entonces solo la usaban los manejadores con su estado escrito a mano. Al crearla con `ErrorAPI` fallaba con `KeyError`. Se corrigió añadiendo `"SOLICITUD_INVALIDA": 422` a `ESTADOS`.

**Al conectar el modelo real.** Este paso no se modifica. `ClasificadorCNN` debe exponer `id` (texto, nunca `null`), `nombre`, `parametros` (entero), `umbral` y `probabilidades(ventanas)`, que devuelve un vector con un valor por ventana.

### Pruebas confirmadas

| Prueba | Resultado esperado | Resultado |
| --- | --- | --- |
| `pytest tests -q` | 41 passed (28 anteriores y 13 nuevas) | Correcto, tras corregir `ESTADOS` |

### Paso 9B: endpoint, tope y limpieza

`POST /api/v1/analisis` ya está conectado. Valida `modelos`, reserva un cupo, lee y valida el CSV, procesa la señal, clasifica con los modelos pedidos y devuelve `RespuestaAnalisis`. Exige la API key y todo ocurre en memoria.

| Archivo | Cambio |
| --- | --- |
| `app/cupo.py` (nuevo) | `Cupo`: semáforo no bloqueante. Sin cupo libre lanza `SERVICIO_OCUPADO` (503) en vez de hacer cola, y libera el cupo siempre, incluso si el análisis falla. |
| `tests/test_cupo.py` (nuevo) | 5 pruebas, una con 5 hilos reales. |
| `app/config.py` | Ajuste nuevo `max_analisis_simultaneos` (2 por defecto, entre 1 y 16); también en `.env.example`. |
| `app/main.py` | `POST /api/v1/analisis`; el `Cupo` se crea una vez en `lifespan`; se borró la ruta temporal `_prueba_archivo`. |

**Decisiones.** `modelos` se valida antes de reservar cupo, porque es barato. El archivo se lee dentro del cupo, así que hay como máximo N archivos en memoria a la vez. El log de cada análisis dice solo el `id_analisis`, los latidos evaluados y los milisegundos, nunca el nombre ni el contenido del archivo. Los errores de validación no escriben log propio, porque el registro de acceso de uvicorn ya anota su código HTTP. `SERVICIO_OCUPADO` no lleva cabecera `Retry-After`, porque `ErrorAPI` no transporta cabeceras: el frontend debe reintentar. El límite de 5 MB se aplica al leer el archivo; el tope del cuerpo antes de leerlo va en el paso 10.

**Al conectar el modelo real.** El endpoint no se modifica: lee `app.state.modelos`, que `registro.py` llena con la simulada o con `ClasificadorCNN`. Conviene ajustar `MAX_ANALISIS_SIMULTANEOS` al número de núcleos del servidor, junto con `TORCH_HILOS`. Verificación pendiente: con el modelo real, `ejemplo_ecg.csv` debe dar 8 latidos anormales (los 52 latidos ya se reproducen); con la CNN simulada salen 6 y no son comparables.

### Pruebas confirmadas (9B)

| Prueba | Resultado esperado | Resultado |
| --- | --- | --- |
| `pytest tests -q` | 46 passed | Correcto, tras renombrar `test_cupo.py` |
| `/analisis` sin clave | 401 `NO_AUTORIZADO` | Correcto |
| `ejemplo_ecg.csv` completo | 200; 52 detectados, 51 evaluados, 1 descartado, FC 104,6; 51 latidos de la CNN simulada; QCNN `no_disponible` | Correcto |
| `modelos=cnn` | 200; el bloque `qcnn` dice «No se solicitó este modelo en la petición» | Correcto |
| `modelos=xyz` | 422 `SOLICITUD_INVALIDA` | Correcto |
| `malo.csv` y `plano.csv` por `/analisis` | 400 `COLUMNAS_INVALIDAS` y 422 `SIN_LATIDOS_SUFICIENTES` | Correcto |
| Ruta `_prueba_archivo` | 404 | Correcto |
| Log de cada análisis | Solo id, latidos y milisegundos | Correcto |
| 5 peticiones a la vez con tope 2 (archivo de 5 minutos) | 2 con 200 y 3 con 503 | Correcto: 200, 200, 503, 503, 503 |

## Paso 10: capas de red (completado)

El paso 10 va en dos partes. La 10A añade tres capas que envuelven toda la app: tamaño del cuerpo, CORS estricto y hosts permitidos. La 10B añade el límite de peticiones por IP.

| Archivo | Cambio |
| --- | --- |
| `app/limite_cuerpo.py` (nuevo) | Middleware ASGI `LimiteTamanoCuerpo`. Responde 413 `ARCHIVO_DEMASIADO_GRANDE` si `Content-Length` supera 5 MB más 64 KB de holgura del multipart, o si un envío por trozos se pasa de ese tamaño. La app ni lo procesa. |
| `tests/test_limite_cuerpo.py` (nuevo) | 6 pruebas con peticiones simuladas. |
| `app/config.py` | Ajuste nuevo `hosts_permitidos` (por defecto `localhost,127.0.0.1`) y propiedad `lista_hosts`; también en `.env.example`. |
| `app/main.py` | Las tres capas, con `add_middleware`. |

| Orden de entrada | Capa | Qué hace |
| --- | --- | --- |
| 1 | Hosts permitidos | Rechaza con 400 las peticiones cuyo `Host` no está en `HOSTS_PERMITIDOS`. |
| 2 | CORS | Solo el origen de `ORIGENES_PERMITIDOS` puede leer respuestas desde un navegador. Métodos GET y POST, cabecera `X-API-Key`, sin credenciales. |
| 3 | Tamaño del cuerpo | Corta con 413 antes de procesar nada. |
| 4 | La app | Autenticación, validación y análisis. |

**Decisiones.** CORS protege a los usuarios en el navegador, pero no es un control de acceso: `curl` y otros clientes que no son navegadores no se bloquean. Ese control es la API key. CORS va por fuera de la capa de tamaño para que la respuesta 413 también lleve sus cabeceras; si no, el navegador ocultaría el error al frontend. La capa de hosts responde texto plano (`Invalid host header`), sin la forma de error del contrato, pero el frontend no la ve en uso normal. El conteo exacto de 5 MB sobre el archivo sigue en `app/archivos.py`.

**Al desplegar.** En el `.env` del servidor, `ORIGENES_PERMITIDOS` debe llevar la URL real del frontend, sin barra final, y `HOSTS_PERMITIDOS` el dominio real de la API. Es parte del paso 11.

### Pruebas confirmadas (10A)

| Prueba | Resultado esperado | Resultado |
| --- | --- | --- |
| `pytest tests -q` | 52 passed | Correcto |
| `/salud` con `Origin` permitido | 200 y `access-control-allow-origin` | Correcto |
| `/salud` con `Origin` ajeno | 200 sin cabeceras CORS | Correcto |
| Preflight `OPTIONS` a `/analisis` | 200 con métodos GET y POST y la cabecera `X-API-Key` | Correcto |
| `Host` falso | 400 `Invalid host header` | Correcto |
| Archivo de 6 MB | 413 `ARCHIVO_DEMASIADO_GRANDE`, con la cabecera CORS | Correcto |
| Análisis normal de `ejemplo_ecg.csv` | 200, 52 detectados y 51 evaluados | Correcto |

### Paso 10B: límite de peticiones por IP

Cada IP puede hacer **30 peticiones por minuto** a `/modelos` y `/analisis`. La 31.ª recibe `429 DEMASIADAS_PETICIONES` con `detalle.reintentar_en_s`. `/salud` queda sin límite porque es pública.

| Archivo | Cambio |
| --- | --- |
| `app/limitador.py` (nuevo) | Clase `LimitadorPorIP`: ventana deslizante de 60 s, con candado para hilos y limpieza de IPs inactivas. Todo en memoria. |
| `tests/test_limitador.py` (nuevo) | 9 pruebas con un reloj falso, incluida una con 12 hilos a la vez. |
| `app/config.py` | Ajuste `limite_peticiones_por_minuto` (por defecto 30; también en `.env.example`). |
| `app/main.py` | El limitador se crea en `lifespan`. La función `limitar_peticiones` va después de los middlewares y antes de las rutas, y las dos rutas protegidas usan `dependencies=[Depends(limitar_peticiones), Depends(exigir_api_key)]`. |

**Decisiones.** El límite va **antes** de la API key, para que también frene los intentos con clave incorrecta. Las peticiones rechazadas no se cuentan, así que insistir no alarga el bloqueo. Los contadores viven en la memoria del proceso: al reiniciar se borran y no se guarda nada en disco.

### Pruebas confirmadas (10B)

| Prueba | Resultado esperado | Resultado |
| --- | --- | --- |
| `pytest tests -q` | 61 passed | Correcto |
| 35 peticiones con clave incorrecta | 30 con 401 y 5 con 429 | Correcto |
| Cuerpo del 429 | `DEMASIADAS_PETICIONES` con `reintentar_en_s` | Correcto |
| `/salud` mientras hay bloqueo | 200 | Correcto |
| Petición válida tras esperar 65 s | 200 | Correcto |

**Al desplegar (paso 11).** Detrás de un proxy HTTPS, `request.client.host` sería la IP del proxy y todos compartirían un contador: hay que arrancar uvicorn con `--proxy-headers --forwarded-allow-ips=<IP del proxy>` (nunca `*`). Con varios procesos cada uno cuenta aparte, así que la demo usa uno solo. `LIMITE_PETICIONES_POR_MINUTO` se ajusta si varias personas comparten red. El código `DEMASIADAS_PETICIONES` (429, con `detalle.reintentar_en_s`) se suma a los códigos extra que hay que avisar al responsable del frontend.

## Paso 10C: CNN y QCNN listas para conectar (completado)

**Hallazgo.** Al auditar el backend, la respuesta del contrato ya tenía el hueco de la QCNN (`no_disponible`), pero faltaba casi todo lo demás. El adaptador real de la CNN solo existía como plantilla en este informe, y la QCNN no tenía adaptador, maqueta, ajustes ni línea en el registro. Además, el repo `qcnn_ecg` todavía no contiene código de la QCNN: solo `pennylane` en las dependencias, y `modelos/__init__.py` dice «la QCNN en el Ciclo 3».

El paso 10C deja a los dos modelos con la misma estructura: un adaptador, un ajuste de modo y una línea en el registro. El endpoint ya ejecutaba cualquier modelo cargado, así que no se tocó.

| Archivo | Cambio |
| --- | --- |
| `app/config.py` | Cuatro ajustes nuevos: `ruta_modelo_cnn`, `torch_hilos`, `qcnn_modo` (`desactivada`, `maqueta` o `real`) y `ruta_modelo_qcnn`. |
| `app/modelos/cnn.py` (nuevo) | `ClasificadorCNN`: carga `cnn1d_final.pt` reutilizando `qcnn_ecg`. Lista, sin activar. |
| `app/modelos/qcnn.py` (nuevo) | `ClasificadorQCNN`: el único archivo que hay que rellenar cuando exista la QCNN. Tiene dos TODO y documenta el contrato de entrada y salida. |
| `app/modelos/simulado.py` | Clase `QCNNMaqueta`: probabilidades inventadas para probar el flujo de dos modelos. |
| `app/modelos/registro.py` | Carga la CNN según `MODELO_SIMULADO` y la QCNN según `QCNN_MODO`. |
| `tests/test_registro.py` (nuevo) | 7 pruebas: modos de la QCNN, contrato de la maqueta y comparabilidad de los latidos. |
| `.env.example` | `QCNN_MODO=desactivada` y `TORCH_HILOS=4`. |

**Decisiones.** `QCNN_MODO` vale `desactivada` por defecto, así que el comportamiento anterior no cambia. Se llama `maqueta` y no `simulada` porque en el contrato la QCNN real también se llama «QCNN (simulada)»: un circuito cuántico ejecutado en un simulador. Si alguien pone `QCNN_MODO=real` antes de completar el adaptador, el servidor no arranca y el error indica el archivo que falta, en vez de que la QCNN desaparezca en silencio.

### Pruebas confirmadas (10C)

| Prueba | Resultado esperado | Resultado |
| --- | --- | --- |
| `pytest tests -q` | 68 passed | Correcto |
| `QCNN_MODO=maqueta`, `/salud` | `modelos_cargados` con `cnn` y `qcnn` | Correcto |
| `QCNN_MODO=maqueta`, análisis de `ejemplo_ecg.csv` | Dos bloques `ok`, 51 latidos cada uno, con los mismos `indice` y `tiempo_s` | Correcto |
| `QCNN_MODO=maqueta`, `/modelos` | QCNN con `disponible: true` e id `qcnn-maqueta` | Correcto |
| `QCNN_MODO=desactivada` | `/salud` solo con `cnn`; QCNN en `no_disponible` con su mensaje | Correcto |

**Riesgo a hablar con el equipo de la QCNN.** La CNN tarda unos 12 ms por análisis, pero simular un circuito cuántico por latido puede ser mucho más lento. Un archivo de 5 min tiene cientos de latidos y el contrato promete menos de 1 s. Cuando se conozca la velocidad real habrá que decidir un límite de latidos o de tiempo para la QCNN: dos análisis lentos bastarían para ocupar los dos cupos.

## Paso 11: pruebas de seguridad y plan de despliegue (despliegue pendiente)

Este paso tiene dos partes. La **11A** son pruebas automáticas de seguridad, ya escritas y por ejecutar. La **11B** es el plan de despliegue: queda explicado aquí, pero **todavía no se ejecuta**.

### 11A: pruebas de seguridad del API (por ejecutar)

Convierten en una suite repetible lo que hasta ahora se probó a mano con `curl`. `TestClient` de FastAPI arranca la app completa dentro de `pytest` y le hace peticiones HTTP sin abrir un servidor.

| Archivo | Qué contiene |
| --- | --- |
| `pyproject.toml` y `uv.lock` | Dependencia de desarrollo `httpx` (`uv add --dev httpx`), que `TestClient` necesita. |
| `tests/conftest.py` (nuevo) | Fija las variables de entorno antes de importar la app: clave de pruebas, modelo simulado, QCNN desactivada y los orígenes y hosts de prueba. Así las pruebas nunca usan la clave real ni cargan modelos reales. |
| `tests/test_seguridad_http.py` (nuevo) | 14 casos, detallados abajo. |

| Qué se comprueba | Resultado esperado |
| --- | --- |
| `/salud` sin clave | 200: es pública |
| `/modelos` sin clave y con clave incorrecta | 401 `NO_AUTORIZADO` |
| `/modelos` con la clave correcta | 200 |
| `/analisis` sin clave | 401 `NO_AUTORIZADO` |
| Respuestas de error | No contienen la clave |
| `Host` no permitido | 400 |
| CORS | Solo el origen configurado recibe `access-control-allow-origin` |
| Cuerpo de 6 MB | 413 `ARCHIVO_DEMASIADO_GRANDE` |
| Límite por IP (3 peticiones) | 401, 401, 401 y 429 con `reintentar_en_s`; `/salud` sigue en 200 |
| `modelos=cnn,gpt` | 422 `SOLICITUD_INVALIDA` |
| Ruta inexistente | 404 con el formato de error del contrato |
| Análisis de `ejemplo_ecg.csv` | 200, con la advertencia, la CNN en `ok` y la QCNN en `no_disponible` |
| Un modelo que lanza un error con un dato «privado» | 200 con `estado: error` y el dato no aparece en la respuesta |

**Estado.** Escritas y revisadas solo en sintaxis; **aún no se han ejecutado**, porque el entorno donde se prepararon no tiene FastAPI. Se ejecutan con `uv run python -m pytest tests -q`; el resultado esperado es **82 passed** (68 anteriores más 14 casos). Si alguna falla, hay que revisar si el error está en la expectativa de la prueba o en un hueco real del backend, y no ajustarla a ciegas.

### 11B: plan de despliegue (documentado, no ejecutado)

**Decisión.** Para la demostración temporal, la API correrá en el PC y se expondrá con un **Cloudflare Quick Tunnel**. No pide cuenta ni dominio, da HTTPS y la URL desaparece al cerrar el túnel, que es lo que necesita una demo temporal. Según la documentación de Cloudflare tiene límites: no garantiza disponibilidad, admite hasta 200 peticiones en vuelo (las demás reciben 429), la URL cambia en cada ejecución y no admite eventos SSE. La restricción por correo (`--allowed-mail`) exige un navegador interactivo, así que **no sirve** para una API que consume React.

```text
Navegador (React) --HTTPS--> Cloudflare --> cloudflared (en el PC) --HTTP--> uvicorn 127.0.0.1:8000
```

| Paso | Acción |
| --- | --- |
| 1 | Preparar el `backend/.env` de producción (tabla de abajo). |
| 2 | Arrancar el servidor **sin** `--reload`: `uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --proxy-headers --forwarded-allow-ips=127.0.0.1` |
| 3 | En otra terminal, abrir el túnel: `cloudflared tunnel --url http://localhost:8000`. Imprime una URL `https://….trycloudflare.com`. |
| 4 | Pasar esa URL al frontend como base de la API. |
| 5 | Verificar a través del túnel: `/salud` en 200; `/modelos` sin clave en 401; un análisis con clave en 200; y la prueba de IPs de abajo. |
| 6 | Al terminar, cerrar el túnel y el servidor con Ctrl+C. La URL deja de funcionar. |

**Por qué esos parámetros de arranque.** `--host 127.0.0.1` hace que el servidor solo sea alcanzable desde el propio PC, es decir, por el túnel. `--proxy-headers` con `--forwarded-allow-ips=127.0.0.1` es lo que hace que el límite por IP funcione: sin eso, `cloudflared` conecta desde `127.0.0.1` y todos los usuarios compartirían un solo contador. Con esos parámetros, uvicorn solo cree en `X-Forwarded-For` si la conexión viene de `127.0.0.1` y toma la dirección más a la derecha que no sea de confianza, que es la que añade Cloudflare. Nunca se usa `*` en `--forwarded-allow-ips`: con `*` se toma la primera dirección de la lista, que el cliente puede falsear.

| Variable del `.env` de producción | Valor |
| --- | --- |
| `API_KEY` | Una clave **nueva** (`uv run python -c "import secrets; print(secrets.token_urlsafe(32))"`), distinta de la de desarrollo, guardada fuera del repo. |
| `DOCS_HABILITADOS` | `false`: sin `/docs`, `/redoc` ni `/openapi.json`, y los errores sin trazas. |
| `MODELO_SIMULADO` | `false` solo si la CNN real ya está conectada (pasos D y E). Si sigue en `true`, la demo muestra resultados simulados y el nombre del modelo lo dice. |
| `QCNN_MODO` | `desactivada`. No usar `maqueta` en una demo pública: mostraría resultados inventados como si fueran de la QCNN. |
| `ORIGENES_PERMITIDOS` | La URL exacta del frontend, sin barra final. |
| `HOSTS_PERMITIDOS` | `localhost,127.0.0.1,*.trycloudflare.com`. El comodín es necesario porque la URL del túnel cambia en cada ejecución. |
| `MAX_ANALISIS_SIMULTANEOS` y `TORCH_HILOS` | Según los núcleos del PC. |
| `LIMITE_PETICIONES_POR_MINUTO` | 30 por defecto; subir si varias personas comparten red. |

**Prueba de que el límite separa a los usuarios.** Con el túnel abierto, lanzar la ráfaga de 35 peticiones con clave incorrecta desde el PC (30 con 401 y 5 con 429). Mientras el PC está bloqueado, hacer una petición válida desde un teléfono con datos móviles: debe dar 200. Si da 429, el límite está contando a todos como una sola IP y hay que revisar `--proxy-headers`.

**Riesgos conocidos.**

| Riesgo | Efecto | Mitigación |
| --- | --- | --- |
| El PC se suspende o se apaga | La API se cae | Conectarlo a la corriente y desactivar la suspensión durante la demo |
| La URL cambia en cada ejecución | El frontend apunta a una URL muerta | Pasar la URL nueva al frontend cada vez; los hosts ya lo toleran por el comodín |
| Cualquiera con la URL puede llamar a la API | Intentos de abuso | Clave, límite por IP, hosts permitidos, tope de tamaño y de análisis simultáneos |
| La clave queda visible en el navegador | No es una defensa fuerte | El límite por IP y los topes son la defensa real; cambiar la clave al terminar la demo |
| Cloudflare responde 429 con más de 200 peticiones en vuelo | Ese 429 no trae el formato de error del contrato | Improbable a esta escala; el frontend debe tolerar una respuesta que no sea JSON |

### Lista previa al despliegue (lo que hay que hacer antes)

| # | Tarea | Estado |
| --- | --- | --- |
| 1 | Ejecutar las pruebas 11A y obtener 82 passed | Pendiente |
| 2 | Confirmar con quien redactó el contrato las interpretaciones abiertas (ver «Cambio al contrato que hay que comunicar») | Pendiente |
| 3 | Avisar al responsable del frontend: enviar `X-API-Key` en `/modelos` y `/analisis`, manejar los códigos nuevos (`NO_AUTORIZADO` 401, `DEMASIADAS_PETICIONES` 429 con `reintentar_en_s`, `SERVICIO_OCUPADO` 503 y `SOLICITUD_INVALIDA`) y darnos la URL exacta del frontend | Pendiente |
| 4 | Decidir qué se muestra en la demo: CNN real (pasos D y E) o simulada, con la QCNN desactivada | Pendiente |
| 5 | Decidir las métricas que publica `GET /modelos` | Pendiente |
| 6 | Generar la API key de producción y entregársela al frontend por un canal privado | Pendiente |
| 7 | Subir el código en la rama `backend` con pull request a `main`, comprobando que `.env` no aparece en `git status` | Pendiente |
| 8 | Endurecimiento opcional: en `limitar_peticiones`, usar `"desconocida"` también cuando `request.client.host` sea `None`, algo que pasa si todas las direcciones de `X-Forwarded-For` son de confianza | Opcional |
| 9 | Instalar `cloudflared` y comprobar que arranca | Pendiente |
| 10 | Conectar el PC a la corriente y desactivar la suspensión el día de la demo | Pendiente |

## Qué subir, conectar y modificar para conectar el modelo final

Los dos modelos ya tienen su adaptador, su ajuste y su línea en el registro (paso 10C). Conectar un modelo real no toca los endpoints: la CNN solo necesita cambios de configuración (pasos D y E de abajo) y la QCNN necesita además rellenar un archivo (sección «Conectar la QCNN»). Quien lo haga necesita el repo completo, Python 3.12 y `uv`. Todos los comandos se ejecutan desde `backend/`, en PowerShell.

El preprocesamiento de la señal (remuestreo, picos R, filtrado y ventanas) usa código real de `qcnn_ecg` y no depende de los modelos. Solo se simula la clasificación: la CNN con CNNSimulada y la QCNN con QCNNMaqueta o desactivada.

### Resumen: qué se toca al conectar

| Modelo | Acción | Dónde | Cambio |
| --- | --- | --- | --- |
| CNN | Modificar | `backend/.env` del servidor | `MODELO_SIMULADO=false` (paso D) y `TORCH_HILOS` según los núcleos |
| CNN | Modificar | `app/config.py` | Poner `modelo_simulado` en `False` por defecto (paso D) |
| CNN | Subir | `modelos_entrenados/cnn1d_final.pt` y el repo completo | Ver «Subir al servidor» |
| CNN | Decidir | Métricas de `GET /modelos` | Con quien tiene los modelos; después se llenan `entrenado_con` y `metricas_prueba` en `cnn.py` |
| QCNN | Rellenar | `app/modelos/qcnn.py` | Los dos TODO de `ClasificadorQCNN` y el `id` real. Es el único archivo de código |
| QCNN | Modificar | `backend/.env` del servidor | `QCNN_MODO=real`, y `RUTA_MODELO_QCNN` si el archivo no se llama `qcnn_final.pt` |
| QCNN | Subir | Archivo del modelo QCNN y sus dependencias | Por ejemplo `pennylane`; luego `uv sync` en el servidor |
| QCNN | Decidir | Límite de latidos o de tiempo | Cuando se conozca su velocidad (ver el riesgo del paso 10C) |
| Ambos | Ajustar | `backend/.env` del servidor | `MAX_ANALISIS_SIMULTANEOS` según los núcleos y `LIMITE_PETICIONES_POR_MINUTO` según los usuarios |
| Ambos | Ajustar | `backend/.env` del servidor | `ORIGENES_PERMITIDOS` con la URL real del frontend y `HOSTS_PERMITIDOS` con el dominio de la API (paso 11) |
| Ambos | No tocar | `main.py`, `analisis.py`, `esquemas.py`, `errores.py`, `seguridad.py`, `cupo.py`, `limitador.py` y los endpoints | La interfaz `Clasificador` los aísla de los modelos |

### Antes de empezar

- [ ] El paso 3 funciona con el modelo simulado: `/api/v1/salud` muestra `["cnn"]`.
- [ ] Existe `modelos_entrenados/cnn1d_final.pt`. Ya está en el repo (53 KB) y lleva el umbral dentro.
- [ ] `uv run python -c "import qcnn_ecg; print(qcnn_ecg.__version__)"` imprime `0.1.0`.
- [ ] Quien tiene los modelos confirmó que el `.pt` conserva las claves `estado`, `config_modelo` y `umbral`.

### Paso A (hecho en el 10C). Cuatro ajustes en `app/config.py`

Dentro de la clase `Ajustes`:

```python
    ruta_modelo_cnn: Path = DIR_BACKEND.parent / "modelos_entrenados" / "cnn1d_final.pt"
    torch_hilos: int = Field(default=4, ge=1, le=64)
    qcnn_modo: Literal["desactivada", "maqueta", "real"] = "desactivada"
    ruta_modelo_qcnn: Path = DIR_BACKEND.parent / "modelos_entrenados" / "qcnn_final.pt"
```

Arriba del archivo va también `from typing import Literal`. `DIR_BACKEND.parent` es la raíz del repo. Los hilos de PyTorch los pide el contrato y se ajustan según el servidor. `QCNN_MODO` vale `desactivada`, `maqueta` o `real`. Todos los valores se pueden sobrescribir en el `.env` con `RUTA_MODELO_CNN`, `TORCH_HILOS`, `QCNN_MODO` y `RUTA_MODELO_QCNN`.

### Paso B (hecho en el 10C). Crear `app/modelos/cnn.py`

```python
from pathlib import Path

import numpy as np
from qcnn_ecg.entrenar import cargar_modelo
from qcnn_ecg.evaluacion import predecir_probabilidades
from qcnn_ecg.modelos import contar_parametros

from app.modelos.base import Clasificador


class ClasificadorCNN(Clasificador):
    clave = "cnn"
    id = "cnn1d-base-v1"
    nombre = "CNN 1-D"
    entrenado_con = "MIT-BIH, DS1 (22 pacientes)"
    # Valores del contrato (promedio de 3 semillas). Confirmar con el equipo de modelos.
    metricas_prueba = {"f1": 0.61, "exactitud": 0.79, "auc_roc": 0.84, "pacientes": 21}

    def __init__(self, ruta: Path):
        self._modelo, self.umbral = cargar_modelo(ruta)
        self.parametros = contar_parametros(self._modelo)

    def probabilidades(self, ventanas: np.ndarray) -> np.ndarray:
        return predecir_probabilidades(self._modelo, ventanas)
```

Reutiliza funciones que ya existen en `qcnn_ecg`: `cargar_modelo` devuelve el modelo y su umbral, y `predecir_probabilidades` convierte a tensor y aplica el sigmoide. No se reimplementa nada. Los atributos entrenado\_con y metricas\_prueba alimentan GET /modelos.

### Paso C (hecho en el 10C). Reemplazar `app/modelos/registro.py`

```python
import logging

from app.config import Ajustes
from app.modelos.base import Clasificador
from app.modelos.simulado import CNNSimulada, QCNNMaqueta

log = logging.getLogger("uvicorn.error")


def _cargar_cnn(ajustes: Ajustes) -> Clasificador:
    if ajustes.modelo_simulado:
        log.warning("MODELO SIMULADO activo: las clasificaciones de la CNN NO son reales.")
        return CNNSimulada()
    # Imports aquí dentro: en modo simulado el servidor arranca rápido y sin cargar PyTorch.
    import torch

    from app.modelos.cnn import ClasificadorCNN

    torch.set_num_threads(ajustes.torch_hilos)
    return ClasificadorCNN(ajustes.ruta_modelo_cnn)


def _cargar_qcnn(ajustes: Ajustes) -> Clasificador | None:
    if ajustes.qcnn_modo == "desactivada":
        return None  # el análisis responde "no_disponible" para la QCNN
    if ajustes.qcnn_modo == "maqueta":
        log.warning("QCNN en modo MAQUETA: las clasificaciones de la QCNN NO son reales.")
        return QCNNMaqueta()
    from app.modelos.qcnn import ClasificadorQCNN

    return ClasificadorQCNN(ajustes.ruta_modelo_qcnn)


def cargar_modelos(ajustes: Ajustes) -> dict[str, Clasificador]:
    modelos: dict[str, Clasificador] = {"cnn": _cargar_cnn(ajustes)}
    qcnn = _cargar_qcnn(ajustes)
    if qcnn is not None:
        modelos["qcnn"] = qcnn
    return modelos
```

Las importaciones de `torch` y `ClasificadorCNN` van dentro de`  _cargar_cnn a propósito: en modo simulado el servidor arranca rápido y no carga PyTorch. La CNN y la QCNN se cargan por separado: con QCNN_MODO=desactivada la QCNN no aparece en el registro y el análisis responde no_disponible `.

### Paso D (pendiente: al tener el modelo). Cambiar la configuración de la CNN

- En el `backend/.env` del servidor: `MODELO_SIMULADO=false`.
- En `app/config.py`: cambiar el valor por defecto de `modelo_simulado` a `False`, para que un `.env` olvidado no deje un servidor público con el modelo falso.
- Reiniciar el servidor, porque `--reload` no vigila el `.env`.

### Paso E (pendiente). Verificar la CNN

1. El modelo suelto, desde `backend/`:

```powershell
uv run python -c "from pathlib import Path; from app.modelos.cnn import ClasificadorCNN; m = ClasificadorCNN(Path('../modelos_entrenados/cnn1d_final.pt')); print(m.id, m.parametros, m.umbral)"
```

Debe imprimir `cnn1d-base-v1 11265 0.6149...`.

2. El servidor: `/api/v1/salud` muestra `["cnn"]` y la consola no escribe la advertencia de modelo simulado.
3. Cuando el preprocesamiento esté hecho (pasos 8 y 9), `ejemplo_ecg.csv` debe dar 52 latidos detectados, 51 evaluados y 8 anormales, como en `ejemplo_respuesta.json`. Las probabilidades pueden diferir en el cuarto decimal.

### Si algo falla

| Síntoma | Causa probable | Solución |
| --- | --- | --- |
| `FileNotFoundError` al arrancar | La ruta del `.pt` no existe | Revisar `RUTA_MODELO_CNN` o la ubicación de `modelos_entrenados/` |
| `ModuleNotFoundError: qcnn_ecg` | Falta el repo completo o no se corrió `uv sync` | Correr `uv sync` en la raíz del repo |
| `KeyError` sobre `config_modelo`, `estado` o `umbral` | El `.pt` tiene otro formato | Adaptar `ClasificadorCNN` al formato nuevo |
| `RuntimeError` al cargar los pesos | El `.pt` y el código de `qcnn_ecg/modelos` son de versiones distintas | Actualizar ambos juntos |
| `modelo.id` sigue diciendo «simulada» | `MODELO_SIMULADO` sigue en `true` o no se reinició el servidor | Revisar el `.env` y reiniciar |

### Subir al servidor

| Qué | Dónde | Nota |
| --- | --- | --- |
| Código del backend: `backend/`, `.env.example`, `pyproject.toml` y `uv.lock` | Rama `backend`, luego pull request a `main` | El `.env` real nunca se sube |
| `modelos_entrenados/cnn1d_final.pt` | Ya está en el repo. El umbral 0.615 va dentro del archivo. | Si se reentrena, hay que reemplazar este archivo |
| Repo completo en el servidor | El paquete `qcnn_ecg` se importa desde `src/` | Python 3.12 y `uv sync` en el servidor |
| `.env` del servidor | Valores reales de producción | `API_KEY` nueva, `ORIGENES_PERMITIDOS` con la URL del frontend, `DOCS_HABILITADOS=false`, `MODELO_SIMULADO=false` |

### Conectar la QCNN (Ciclo 3)

Hoy la QCNN está desactivada (`QCNN_MODO=desactivada`) y el API devuelve `"estado": "no_disponible"` para esa clave. El hueco ya está preparado en el paso 10C; lo que falta es el modelo, porque el repo `qcnn_ecg` todavía no tiene código de la QCNN. Cuando exista, se conecta así, sin tocar los endpoints:

1. **Pedir a quien entrena la QCNN** el archivo del modelo y su código, con lo que dice el docstring de `ClasificadorQCNN`: entrada `(n, 288)` de números decimales, ya filtrada (0.5 a 40 Hz) y con z-score, igual que la CNN; salida `(n,)` con la probabilidad de ANORMAL de cada latido, entre 0 y 1; además el `umbral`, el número de parámetros y un `id` versionado.
2. **Rellenar `app/modelos/qcnn.py`:** cargar el modelo en `__init__` (TODO 1), devolver las probabilidades en `probabilidades` (TODO 2) y poner el `id` real. Quitar el `raise NotImplementedError` de ambos métodos.
3. **Subir** el archivo del modelo y las dependencias que necesite, por ejemplo `pennylane`, y correr `uv sync` en el servidor.
4. **Configurar el `.env` del servidor:** `QCNN_MODO=real`, y `RUTA_MODELO_QCNN` si el archivo no es `modelos_entrenados/qcnn_final.pt`. Reiniciar, porque `--reload` no vigila el `.env`.
5. **Verificar:** `/salud` con `cnn` y `qcnn` en `modelos_cargados`; `/modelos` con la QCNN en `disponible: true`; y un análisis de `ejemplo_ecg.csv` con los dos bloques en `ok` y los mismos `indice` y `tiempo_s` por latido.

**Antes de tener el modelo real** se puede probar el flujo completo con `QCNN_MODO=maqueta`. La maqueta no es la QCNN: sus probabilidades son inventadas y la consola lo avisa.

| Síntoma | Causa probable | Solución |
| --- | --- | --- |
| El servidor no arranca con `NotImplementedError` | `QCNN_MODO=real` con el adaptador sin completar | Completar `app/modelos/qcnn.py` o volver a `QCNN_MODO=desactivada` |
| La QCNN sale en `estado: error` en un análisis | El modelo falló con ese archivo; el log solo guarda el tipo de error | Probar `probabilidades` con una ventana suelta y revisar la forma de la salida |
| Los análisis tardan o dan 503 | La QCNN es lenta y ocupa los cupos | Decidir un límite de latidos o de tiempo (ver el riesgo del paso 10C) |
| `QCNN_MODO` no tiene efecto | No se reinició el servidor | Reiniciar |

### Decisión pendiente con quien tiene los modelos

Las métricas que mostrará `GET /modelos` no coinciden entre dos fuentes. El contrato y el README dan F1 0.61, exactitud 0.79 y AUC-ROC 0.84, promedio de 3 semillas. El archivo `modelos_entrenados/cnn1d_final.json` (semilla 0, el modelo que se carga) da F1 0.552, exactitud 0.813 y AUC-ROC 0.813. Hay que decidir cuáles se publican.

## Puntos abiertos

Decisiones y confirmaciones que el backend necesita de otras personas.

| # | Punto | Con quién |
| --- | --- | --- |
| 1 | **Interpretaciones del contrato.** (a) El paso de tiempo constante «< 1 %» se mide como desviación máxima respecto al paso medio. (b) La duración es número de muestras ÷ frecuencia, redondeada a centésimas. (c) El bloque `qcnn` siempre está presente, como `no_disponible` cuando no se pidió o no existe. (d) Un valor desconocido en `modelos` da `SOLICITUD_INVALIDA` (422). (e) Si un modelo falla, el HTTP es 200 y su bloque lleva `estado: error`. | Quien escribió el contrato |
| 2 | **Códigos de error nuevos** (`NO_AUTORIZADO` 401, `DEMASIADAS_PETICIONES` 429, `SERVICIO_OCUPADO` 503 y `SOLICITUD_INVALIDA`), y que el frontend envíe `X-API-Key` en `/modelos` y `/analisis`. | Responsable del frontend |
| 3 | **Métricas de `GET /modelos`.** El contrato da F1 0.61, exactitud 0.79 y AUC-ROC 0.84 (promedio de 3 semillas); el archivo del modelo cargado da F1 0.552, exactitud 0.813 y AUC-ROC 0.813 (semilla 0). | Quien tiene los modelos |
| 4 | **CNN real.** El adaptador está listo y el archivo `cnn1d_final.pt` ya está en el repo. Falta decidir cuándo conectarla (pasos D y E). | Quien tiene los modelos y el líder |
| 5 | **QCNN real.** No existe código de la QCNN en `qcnn_ecg`. Hay que acordar el formato del archivo, la entrada y la salida (ver el docstring de `ClasificadorQCNN`) y su velocidad: simular un circuito por latido puede ser mucho más lento que la CNN. | Quien entrena la QCNN |
| 6 | **Despliegue.** El plan (túnel de Cloudflare desde un PC, con la lista previa) está en el paso 11B. Falta su aprobación. | Líder |
| 7 | **Cita de los datos de ejemplo.** `ejemplo_ecg.csv` es un fragmento de la MIT-BIH Arrhythmia Database (Moody y Mark, 2001, PhysioNet, licencia Open Data Commons Attribution v1.0). Si el ejemplo se publica, hay que citar la fuente. | Líder |

## Estado y próximos pasos

El siguiente paso es ejecutar las pruebas de seguridad (11A) y revisar la lista previa al despliegue del paso 11. El despliegue no se hace todavía: su plan está documentado en el paso 11B. La rama backend se entrega para revisión del líder y no se une a main hasta su aprobación. Para conectar los modelos finales, ver la sección «Qué subir, conectar y modificar».

| Paso | Qué | Estado |
| --- | --- | --- |
| 1 | Entorno y `/salud` | Completado |
| 2 | Configuración y secretos | Completado, 4 de 4 pruebas correctas |
| 3 | Servicio de modelos con CNN simulada | Completado, con el modelo simulado |
| 4 | Esquemas Pydantic y `GET /modelos` | Completado |
| 5 | Manejadores de error globales | Completado |
| 6 | Dependencia de la API key | Completado |
| 7 | Subida y validación del CSV | Completado, 15 pruebas unitarias y 2 pruebas HTTP correctas |
| 8 | Remuestreo, picos R y ventanas | Completado, 28 pruebas unitarias y verificación con el ejemplo (52 detectados, 51 evaluados) |
| 9 | Ensamblar `POST /analisis` | Completado, 46 pruebas unitarias y pruebas HTTP del endpoint |
| 10 | CORS, hosts y rate limiting | Completado, 61 pruebas unitarias y pruebas HTTP de CORS, hosts, 413 y 429 |
| 10C | CNN y QCNN listas para conectar | Completado, 68 pruebas unitarias y pruebas HTTP con la QCNN en maqueta y desactivada |
| 11 | Pruebas de seguridad y despliegue con HTTPS | 11A: pruebas escritas, por ejecutar. 11B: plan documentado; despliegue no realizado |
