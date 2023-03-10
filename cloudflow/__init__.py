from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:  # pragma: no cover
    __version__ = "unknown"
finally:
    del get_distribution, DistributionNotFound


from cloudflow.cloudflow_model import cloudflow_model
from cloudflow.custom_class_mlflow import custom_ml_flow_model
from cloudflow.prepare_env import prepare_env
