from app.modelos.cnn import ClasificadorCNN


class ClasificadorQCNN(ClasificadorCNN):
    """QCNN híbrida (front-end Conv1d + circuito de 8 qubits simulado), ensamble de 5 modelos.

    Se carga y se evalúa igual que la CNN: `cargar_modelo` reconstruye la QCNN desde el
    `config_modelo` guardado y `predecir_probabilidades` pasa las ventanas y los RR.
    """

    clave = "qcnn"
    id = "qcnn-hibrida-v1"
    nombre = "QCNN (simulada)"  # simulada = circuito cuántico ejecutado en un simulador
