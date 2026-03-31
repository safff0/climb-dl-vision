import numpy as np
from catboost import CatBoostClassifier, Pool


class HandcraftedColorClassifier:
    def __init__(self):
        self.model = None
        self.class_names = []

    def fit(self, features: np.ndarray, labels: np.ndarray, class_names: list[str]):
        self.class_names = class_names
        self.model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            loss_function="MultiClass",
            verbose=50,
            auto_class_weights="Balanced",
        )
        self.model.fit(features, labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features).flatten().astype(int)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)

    def save(self, path: str):
        self.model.save_model(path, format="cbm",
                              pool=None)
        metadata_path = path + ".meta.npy"
        np.save(metadata_path, np.array(self.class_names, dtype=object))

    @classmethod
    def load(cls, path: str) -> "HandcraftedColorClassifier":
        obj = cls()
        obj.model = CatBoostClassifier()
        obj.model.load_model(path)
        metadata_path = path + ".meta.npy"
        obj.class_names = np.load(metadata_path, allow_pickle=True).tolist()
        return obj
