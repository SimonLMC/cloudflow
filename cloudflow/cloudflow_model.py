import mlflow
import os
import logging
from cloudflow import init_logger
from cloudflow import custom_class_mlflow


class cloudflow_model():
    def __init__(self, log_level = "INFO"):
        init_logger.init_logger(level=log_level)
        self.model_to_save = custom_class_mlflow.custom_ml_flow_model()
        return 
    
    def save(self, tracking_uri, experiment_id, run_id, predict_function, models):

        artifact_path = os.path.join(tracking_uri, "mlruns", experiment_id, run_id)
        self.model_to_save.save(artifact_path, predict_function, models)
        
    def load_model(self, tracking_uri,experiment_id ,model_id):  

        model_path = os.path.join(tracking_uri, "mlruns", experiment_id, model_id, "artifacts/Model")
        self.model_loaded = mlflow.pyfunc.load_model(model_path)
        
    def predict(self, **params):
        return self.model_loaded.predict(params)