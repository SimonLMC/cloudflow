import os
import sys
import mlflow
import pkgutil
import importlib
import cloudpickle
import logging

from cloudflow import init_logger

init_logger(level="INFO")
  
logging.info("This a my info log message")

class custom_ml_flow_model(mlflow.pyfunc.PythonModel):

    def __init__(self):
        return
    
    def variable_for_value(self, obj):
        """
        Get the choosed name of an object
        obj object : Object for which we need to retrieve the name
        Return object saved name
        """

        for n,v in globals().items():
            if v == obj:
                return n
        return None
        
    def load_context(self, context):
        """
        load all the artifacts associated with the MLflow model to load

        context MLflow object : default argument of the custom MLflow class. Used to import the artifacts loaded by the load_context function
        """
        
        # We need to import those packages inside the function as only the class is pickled with MLflow (not the file). 
        # As the loading environment does not necessarily have these packages pre-imported. We import them here.
        import cloudpickle
        import inspect
        
        list_artifacts_name = []
        
        # loading of all model's associated artifacts
        for name, artifact in context.artifacts.items():
            list_artifacts_name.append(name)
            print("loading {}".format(name))
            with open(artifact, "rb") as f:
                exec("self.{} =  cloudpickle.load(f)".format(name))
            print("loading {} --> DONE".format(name))
                

        # retrieve spec (args mostly) of the user defined function
        signature = inspect.signature(self.__predict__)

        # retrieve default values of the user defined function
        self.arg_default_value = {
                                k: v.default
                                for k, v in signature.parameters.items()
                                if v.default is not inspect.Parameter.empty
                                }
        #retrive all arguments of the user defined function
        list_args_predict = [k for k, v in signature.parameters.items()]

        dict_args___predict__ = {}
        
        # transform the user defined function to adapt the arguments of the function to the specificity of MLflow
        for arg in list_args_predict:
            if arg in list_artifacts_name:
                dict_args___predict__[arg] = "self.{}".format(arg)
            else :
                dict_args___predict__[arg] = "args[\"{}\"]".format(arg)
                
        self.dict_args___predict__ = dict_args___predict__
        
    def register_all_function_by_value(self):
        """
        Register all necessary files and function usefull to the proper functioning of the predict_function
        """    
        
        sub_folder_names = [(x) for x in os.listdir() if os.path.isdir(x)]
    
        # for loop on all files of the project to find all necessary modules and function for the user defined function to save
        for fold_name, folder in [(None, os.getcwd())] + [(x, os.path.join(os.getcwd(), x)) for x in os.listdir() if os.path.isdir(x)]:
            if ".ipynb_" not in folder:
                for module in [name for _, name, _ in pkgutil.iter_modules([folder]) if name not in sub_folder_names]:
                    print(module)
                    # if module is in a subfolder, we need to add the subfolder to the module name
                    full_module = module if fold_name is None else fold_name + "." + module
                    module_spec = importlib.util.spec_from_file_location(
                                                            full_module,os.path.join(folder, "{}.py".format(module))
                                                            )
                    module = importlib.util.module_from_spec(module_spec)
                    sys.modules[module_spec.name] = module
                    module_spec.loader.exec_module(module)
                    cloudpickle.register_pickle_by_value(module)
            
    def pickle_artifacts(self, artifact_path, predict_function, models):
        """
        pickle the given object (or function)

        artifact_path string: Path in which to save artifacts (function predict and models)
        predict_function function: function associated to the models, to save
        models pickable object: any pickable object associated with the predict_function
        """


        artifacts = {}
        
        name_pred = "__predict__"
        path_pred = os.path.join(artifact_path, '{}.pkl'.format(name_pred))
        artifacts[name_pred] = path_pred
        
        with open(path_pred, "wb") as f:
            cloudpickle.dump(predict_function, f)

        for model in models:
            name_model = self.variable_for_value(model)
            path_model = os.path.join(artifact_path, "{}.pkl".format(name_model))
            artifacts[name_model] = path_model
            
            with open(path_model, "wb") as f:
                cloudpickle.dump(model, f)
            
        return artifacts
        
                
    def save(self, artifact_path, predict_function, models):
        """
        Use to save the user defined function (and any associated models)

        artifact_path string: Path in which to save artifacts (function predict and models)
        predict_function function: function associated to the models, to save
        models pickable object: any pickable object associated with the predict_function

        """
        
        self.register_all_function_by_value()
        
        artifacts = self.pickle_artifacts(artifact_path, predict_function, models)

        # default MLflow functoin to save a model        
        mlflow.pyfunc.log_model(
            artifact_path = "Model",
            python_model   =  self,
            artifacts      =  artifacts)
        
        print(artifacts)
        
        # remove artifacts from the temporary folder
        for name, _ in artifacts.items():
            os.remove(artifacts[name])
                            
    def predict(self, context, args): 
        """
        Uses the saved function (and any associated models) defined by the user at the time of saving 

        context MLflow object : default argument of the custom MLflow class. Used to import the artifacts loaded by the load_context function
        args dictionnary: dictionnary that contain the differents arguments used by the predict function
    
        Return the associated return from the user defined saved function
        """
        
        # retrieve all default value if no value is given by the user in the predict
        for arg in self.arg_default_value:
            if arg not in args.keys():
                args[arg] = self.arg_default_value[arg]
        
        dict_args = {}
        
        for name_arg, value_arg in self.dict_args___predict__.items():
            dict_args[name_arg] = eval(value_arg)  
            
        return self.__predict__(**dict_args)
