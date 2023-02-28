import mlflow
import os
import logging
from cloudflow import init_logger
from cloudflow import custom_class_mlflow

class cloudflow_model():
    def __init__(self, log_level = "ERROR"):
        """
        Initializes the CloudFlow model object.

        Args:
            log_level (str): The log level for the logger. Defaults to "ERROR".

        Returns:
            None.

        This function initializes the CloudFlow model object and sets up the logger.
        """
        init_logger.init_logger(level=log_level)
        self.model_to_save = custom_class_mlflow.custom_ml_flow_model()
        logging.info('Initialized cloudflow_model with log_level %s', log_level)
    
    def save(self, tracking_uri, experiment_id, run_id, predict_function, models):
        """
        Saves the model and its artifacts to a specified location.

        Args:
            tracking_uri (str): The URI of the tracking server.
            experiment_id (str): The ID of the experiment in which the run is created.
            run_id (str): The ID of the run.
            predict_function (function): The function to use for prediction.
            models (list): A list of trained models to save.

        Returns:
            None.

        This function saves the model and its artifacts to a specified location using the MLflow tracking server.
        """
        artifact_path = os.path.join(tracking_uri, "mlruns", experiment_id, run_id)
        self.model_to_save.save(artifact_path, predict_function, models, run_id)
        logging.info('Model saved at %s', artifact_path)
    
    def load_model(self, tracking_uri,experiment_id ,model_id):  
        """
        Loads the specified model from the tracking server.

        Args:
            tracking_uri (str): The URI of the tracking server.
            experiment_id (str): The ID of the experiment in which the run is created.
            model_id (str): The ID of the model to load.

        Returns:
            None.

        This function loads the specified model from the MLflow tracking server and sets the model as the 
        `model_loaded` attribute of the object.
        """
        model_path = os.path.join(tracking_uri, "mlruns", experiment_id, model_id, "artifacts/Model")
        self.model_loaded = mlflow.pyfunc.load_model(model_path)
        logging.info('Model loaded from %s', model_path)
        
    def predict(self, **params):
        """
        Predicts using the loaded model.

        Args:
            params: Keyword arguments to use as input for prediction.

        Returns:
            The prediction result.

        This function uses the `model_loaded` attribute of the object to make predictions and returns the result.
        """
        result = self.model_loaded.predict(params)
        logging.info('Model predicted %s', result)
        return result
