import os

# Estas variables se fijan ANTES de importar la app, porque main.py lee la
# configuración al importarse. Las variables de entorno tienen prioridad sobre
# el .env, así que las pruebas nunca usan tu clave real ni cargan modelos reales.
os.environ["API_KEY"] = "clave-solo-para-pruebas-0123456789"
os.environ["MODELO_SIMULADO"] = "true"
os.environ["QCNN_MODO"] = "desactivada"
os.environ["ORIGENES_PERMITIDOS"] = "http://localhost:5173"
os.environ["HOSTS_PERMITIDOS"] = "localhost,127.0.0.1"
os.environ["LIMITE_PETICIONES_POR_MINUTO"] = "1000"