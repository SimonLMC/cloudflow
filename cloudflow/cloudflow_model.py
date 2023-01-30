import mlflow
import os

from cloudflow.custom_class_mlflow import custom_ml_flow_model


class cloudflow_model():
    def __init__(self):
        self.model_to_save = custom_ml_flow_model()
        return 
    
    def save(self, artifact_path, predict_function, models):
        self.model_to_save.save(artifact_path, predict_function, models)
        
    def load_model(self, folder_path ,model_id):        
        self.model_loaded = mlflow.pyfunc.load_model(os.path.join(folder_path ,model_id, "artifacts/Model"))
        
    def predict(self, **params):
        return self.model_loaded.predict(params)