import mlflow
import os

import os
import pkgutil
import importlib
import sys
import mlflow
import cloudpickle

class custom_ml_flow_model(mlflow.pyfunc.PythonModel):

    def __init__(self):
        return
    
    
    def variable_for_value(self, value):
        for n,v in globals().items():
            if v == value:
                return n
        return None
        
    def load_context(self, context):
        
        import cloudpickle
        import inspect
        
        list_artifacts_name = []
        
        for name, artifact in context.artifacts.items():
            list_artifacts_name.append(name)
            print("loading {}".format(name))
            with open(artifact, "rb") as f:
                exec("self.{} =  cloudpickle.load(f)".format(name))
            print("chargé")
                

        signature = inspect.signature(self.__predict__)

        self.arg_default_value = {
                                k: v.default
                                for k, v in signature.parameters.items()
                                if v.default is not inspect.Parameter.empty
                                }

        list_args_predict = [k for k, v in signature.parameters.items()]

        dict_args___predict__ = {}
        
        for arg in list_args_predict:
            if arg in list_artifacts_name:
                dict_args___predict__[arg] = "self.{}".format(arg)
            else :
                dict_args___predict__[arg] = "args[\"{}\"]".format(arg)
                
        self.dict_args___predict__ = dict_args___predict__
        
    def register_all_function_by_value(self):    
        
        sub_folder_names = [(x) for x in os.listdir() if os.path.isdir(x)]
    
        for fold_name, folder in [(None, os.getcwd())] + [(x, os.path.join(os.getcwd(), x)) for x in os.listdir() if os.path.isdir(x)]:
            if ".ipynb_" not in folder:
                for module in [name for _, name, _ in pkgutil.iter_modules([folder]) if name not in sub_folder_names]:
                    print(module)
                    full_module = module if fold_name is None else fold_name + "." + module
                    module_spec = importlib.util.spec_from_file_location(full_module,os.path.join(folder, "{}.py".format(module)))
                    module = importlib.util.module_from_spec(module_spec)
                    sys.modules[module_spec.name] = module
                    module_spec.loader.exec_module(module)
                    cloudpickle.register_pickle_by_value(module)
            
    def pickle_artifacts(self, artifact_path, predict_function, models):
        artifacts = {}
        
        name_pred = "__predict__"
        path_pred = os.path.join(artifact_path, '{}.pkl'.format(name_pred))
        with open(path_pred, "wb") as f:
            cloudpickle.dump(predict_function, f)
            
            artifacts[name_pred] = path_pred

        for model in models:
            name = self.variable_for_value(model)
            path = os.path.join(artifact_path, "{}.pkl".format(name))
            with open(path, "wb") as f:
                cloudpickle.dump(model, f)
                
            artifacts[name] = path
            
        return artifacts
        
                
    def save(self, artifact_path, predict_function, models):
        
        self.register_all_function_by_value()
        
        artifacts = self.pickle_artifacts(artifact_path, predict_function, models)
                
        mlflow.pyfunc.log_model(
            artifact_path = "Model",
            python_model   =  self,
            artifacts      =  artifacts)
        
        print(artifacts)
        
        for name, inter_model in artifacts.items():
            os.remove(artifacts[name])
                            
    def predict(self, context, args): 
        """
        make and combine the prediction of all the differents models on the differents scoring tables
    
        params_scoring dictionnary: dictionnary that contain all the scoring tables on which models will...
        ... make predictions. It have to be a dictionnary as the predict function of mlflow only take a single argument.
    
        Return Pandas dataframe with all the fraud risk score (and few others informations) on the remise batch
        Return interpretation_remises pandas dataframe with the interpretation for the remises model
        Return interpretation_client pandas dataframe with the interpretation for the clients models
        """
        
        for arg in self.arg_default_value:
            if arg not in args.keys():
                args[arg] = self.arg_default_value[arg]
        
        new_dict = {}
        
        for x, y in self.dict_args___predict__.items():
            new_dict[x] = eval(y)  
            
        return self.__predict__(**new_dict)
