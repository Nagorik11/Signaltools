import os
import json
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class Learner:
    def __init__(self, output_dir: str = "output/"):
        self.output_dir = output_dir
        self.model = KMeans(n_clusters=3) # Ejemplo: Agrupar en 3 tipos de señales

    def collect_features(self) -> np.ndarray:
        """Carga todos los JSON y extrae los glyph_vectors."""
        vectors = []
        for filename in os.listdir(self.output_dir):
            if filename.endswith(".json"):
                with open(os.path.join(self.output_dir, filename), 'r') as f:
                    data = json.load(f)
                    if "glyph" in data:
                        vectors.append(data["glyph"])
        return np.array(vectors)

    def train(self):
        """Entrena el modelo con las señales disponibles."""
        features = self.collect_features()
        if len(features) > 0:
            self.model.fit(features)
            print(f"[+] Modelo entrenado con {len(features)} señales.")
        else:
            print("[-] No hay datos suficientes para entrenar.")

    def predict(self, signal_glyph: list) -> int:
        """Dice a qué 'familia' pertenece una señal nueva."""
        return self.model.predict([signal_glyph])[0]
    

def __main__():
    return 0

if __name__ == "__main__":
    __main__()