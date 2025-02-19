import os
import pandas as pd
import xgboost as xgb
from abc import ABC, abstractmethod
from wearablesinternational.Exceptions import MLModelException

# Abstract MLModel class
class MLModel(ABC):
    @abstractmethod
    def predict(self, df: pd.DataFrame):
        pass


# Stress
class MLModelStress(MLModel):
    def __init__(self, models_folder):
        self._inputs = ["HR", "EDA"]
        if not os.path.exists(os.path.join(models_folder,  "model_stress.xgb")):
            raise MLModelException(301, f"Model type 'model_stress.xgb' not found.")
        else:
            model = xgb.Booster()
            model.load_model(os.path.join(models_folder,  "model_stress.xgb"))
            self.model = model

    def predict(self, df: pd.DataFrame):
        if df.empty:
            raise MLModelException(303, "MLModelStress.predict: The input DataFrame is empty.")
        if not all(col in df.columns for col in ["HR", "EDA"]):
            raise MLModelException(304, "MLModelStress.predict: Required columns are: HR, EDA.")
        hr = df["HR"].tolist()
        eda = df["EDA"].tolist()
        temp = pd.DataFrame({"hr": hr,"eda": eda})
        stress = self.model.predict(xgb.DMatrix(temp[["hr","eda"]]))
        return stress
    
    @property
    def inputs(self):
        return self._inputs


# Attention
class MLModelAttention(MLModel):
    def __init__(self, models_folder):
        self._inputs = ["HR", "EDA"]
        if not os.path.exists(os.path.join(models_folder,  "model_attention.xgb")):
            raise MLModelException(301, f"Model type 'model_attention.xgb' not found.")
        else:
            model = xgb.Booster()
            model.load_model(os.path.join(models_folder,  "model_attention.xgb"))
            self.model = model

    def predict(self, df: pd.DataFrame):
        if df.empty:
            raise MLModelException(303, "MLModelAttention.predict: The input DataFrame is empty.")
        if not all(col in df.columns for col in ["HR", "EDA"]):
            raise MLModelException(304, "MLModelAttention.predict: Required columns are: HR, EDA.")
        hr = df["HR"].tolist()
        eda = df["EDA"].tolist()
        temp = pd.DataFrame({"hr": hr,"eda": eda})
        attention = self.model.predict(xgb.DMatrix(temp[["hr","eda"]]))
        return attention
    
    @property
    def inputs(self):
        return self._inputs


# Valence
class MLModelValence(MLModel):
    def __init__(self, models_folder):
        self._inputs = ["HR", "EDA"]
        if not os.path.exists(os.path.join(models_folder, "model_valence.xgb")):
            raise MLModelException(301, f"Model type 'model_valence.xgb' not found.")
        else:
            model = xgb.Booster()
            model.load_model(os.path.join(models_folder,  "model_valence.xgb"))
            self.model = model

    def predict(self, df: pd.DataFrame):
        if df.empty:
            raise MLModelException(303, "MLModelValence.predict: The input DataFrame is empty.")
        if not all(col in df.columns for col in ["HR", "EDA"]):
            raise MLModelException(304, "MLModelValence.predict: Required columns are: HR, EDA.")
        hr = df["HR"].tolist()
        eda = df["EDA"].tolist()
        temp = pd.DataFrame({"hr": hr,"eda": eda})
        valence = self.model.predict(xgb.DMatrix(temp[["hr","eda"]]))
        return valence
    
    @property
    def inputs(self):
        return self._inputs


# Arousal
class MLModelArousal(MLModel):
    def __init__(self, models_folder):
        self._inputs = ["HR", "EDA"]
        if not os.path.exists(os.path.join(models_folder, "model_arousal.xgb")):
            raise MLModelException(301, f"Model type 'model_arousal.xgb' not found.")
        else:
            model = xgb.Booster()
            model.load_model(os.path.join(models_folder, "model_arousal.xgb"))
            self.model = model

    def predict(self, df: pd.DataFrame):
        if df.empty:
            raise MLModelException(303, "MLModelArousal.predict: The input DataFrame is empty.")
        if not all(col in df.columns for col in ["HR", "EDA"]):
            raise MLModelException(304, "MLModelArousal.predict: Required columns are: HR, EDA.")
        hr = df["HR"].tolist()
        eda = df["EDA"].tolist()
        temp = pd.DataFrame({"hr": hr,"eda": eda})
        arousal = self.model.predict(xgb.DMatrix(temp[["hr","eda"]]))
        return arousal
    
    @property
    def inputs(self):
        return self._inputs


# Abstract MLModel Factory
class MLModelFactory:
    _models = {}
    _models_folder = ""

    @property
    @abstractmethod
    def inputs(self):
        pass

    # note: needs to be instantiated for loading of models to occur
    def __init__(self, models_folder):
        if os.path.exists(models_folder):
            self._models_folder = models_folder
            files = os.listdir(models_folder)
            for file in files:
                if file == "model_stress.xgb":
                    self._models["Stress"] = lambda: MLModelStress(self._models_folder)
                if file == "model_attention.xgb":
                    self._models["Attention"] = lambda: MLModelAttention(self._models_folder)
                if file == "model_arousal.xgb":
                    self._models["Arousal"] = lambda: MLModelArousal(self._models_folder)
                if file == "model_valence.xgb":
                    self._models["Valence"] = lambda: MLModelValence(self._models_folder)

    @staticmethod
    def GetModel(model_type: str) -> MLModel:
        mlmodel_class = MLModelFactory._models.get(model_type)
        if not mlmodel_class:
            raise MLModelException(302, f"Model type '{model_type}' is not implemented.")
        return mlmodel_class()

    @staticmethod
    def ListModels() -> list:
        temp = list(MLModelFactory._models.keys())
        temp.sort()
        return temp