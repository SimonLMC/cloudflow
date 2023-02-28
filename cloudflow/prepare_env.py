import mlflow
import os
import yaml

def prepare_env(tracking_uri,experiment_id):
    """
      Prepares the environment for a MLflow experiment by setting the tracking URI 
      and creating a new experiment with the specified experiment ID if it doesn't already exist. 
      If a new experiment is created, it also updates the meta.yaml file to reflect the new experiment ID and artifact location.

        Parameters:

        tracking_uri (str): The URI for tracking the MLflow experiment.
        experiment_id (str): The ID of the experiment to create or use.
        Returns:

        None.
        Raises:

        None.
    """

    tracking_uri = os.path.join(tracking_uri, "mlruns")

    mlflow.set_tracking_uri(tracking_uri)
    spec = mlflow.set_experiment(experiment_name = experiment_id)
    
    if  experiment_id not in [el.experiment_id for el in mlflow.search_experiments()]:
        os.rename(os.path.join(tracking_uri , spec.experiment_id), os.path.join(tracking_uri,experiment_id))
        with open(os.path.join(tracking_uri ,experiment_id, "meta.yaml"), 'r') as f:
            yaml_content = yaml.load(f,Loader=yaml.loader.SafeLoader)
            yaml_content["artifact_location"] = os.path.join(tracking_uri,experiment_id)
            yaml_content["experiment_id"] = experiment_id
        with open(os.path.join(tracking_uri ,experiment_id, "meta.yaml"), 'w') as f:
            yaml.dump(yaml_content, f)            