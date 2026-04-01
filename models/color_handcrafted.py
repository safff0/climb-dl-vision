import numpy as np
from catboost import CatBoostClassifier


class HandcraftedColorClassifier:
    def __init__(self):
        self.model = None
        self.class_names = []
        self.label_map = {}

    def fit(self, features: np.ndarray, labels: np.ndarray, class_names: list[str]):
        present_indices = sorted(set(labels))
        self.class_names = [class_names[i] for i in present_indices]
        self.label_map = {old: new for new, old in enumerate(present_indices)}

        mapped_labels = np.array([self.label_map[l] for l in labels])

        self.model = CatBoostClassifier(
            iterations=500,
            depth=6,
            learning_rate=0.1,
            loss_function="MultiClass",
            verbose=50,
            auto_class_weights="Balanced",
        )
        self.model.fit(features, mapped_labels)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict(features).flatten().astype(int)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(features)

    def save(self, path: str):
        self.model.save_model(path, format="cbm")
        metadata_path = path + ".meta.npz"
        np.savez(metadata_path,
                 class_names=np.array(self.class_names, dtype=object))

    @classmethod
    def load(cls, path: str) -> "HandcraftedColorClassifier":
        obj = cls()
        obj.model = CatBoostClassifier()
        obj.model.load_model(path)
        metadata_path = path + ".meta.npz"
        data = np.load(metadata_path, allow_pickle=True)
        obj.class_names = data["class_names"].tolist()
        return obj
