import os
import sys
import mlflow
import pkgutil
import importlib
import cloudpickle
import logging
import ast
import importlib_metadata
import re



class custom_ml_flow_model(mlflow.pyfunc.PythonModel):

    def __init__(self):
        """
        Initializes a new instance of the class.
        """
        return


    def get_pip_requirements(self, req_path):
        """
        Reads a file containing pip requirements and returns a set of package names.

        Args:
        req_path (str): The path to the file containing the requirements.

        Returns:
        set: A set containing the names of the packages required by the file.
        """
        with open(req_path) as f:
            reqs =  f.read().splitlines()

        return set(reqs)


    def find_imported_package(self, file_path):
        """
        Parses a Python file and returns the set of packages imported from site-packages.

        Args:
        file_path (str): The path to the Python file.

        Returns:
        set: A set containing the names of the packages imported from site-packages.
        """
        with open(file_path, 'r') as file:
            content = file.read()

        tree = ast.parse(content)

        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    package = alias.name.split('.')[0]
                    if package in sys.modules:
                        module = sys.modules[package]
                        if hasattr(module, '__file__') and 'site-packages' in module.__file__:
                            distrib_name = importlib_metadata.packages_distributions()[package]
                            imports.update(distrib_name)
            elif isinstance(node, ast.ImportFrom):
                package = node.module.split('.')[0]
                if package in sys.modules:
                    module = sys.modules[package]
                    if hasattr(module, '__file__') and 'site-packages' in module.__file__:
                        if node.level == 0:
                            distrib_name = importlib_metadata.packages_distributions()[package]
                            imports.update(distrib_name)

        return imports


    def find_imported_package_versions(self, logged_req, imports):
        """
        Finds the versions of the packages imported from site-packages and adds them to a set.

        Args:
        logged_req (set): The set to which the package versions should be added.
        imports (set): The set of package names.

        Returns:
        set: The updated set containing the package names and versions.
        """
        for package_name in imports:
            package_version = importlib_metadata.version(package_name)
            logged_req.add("{} == {}".format(package_name , package_version))
            
        return logged_req


    def get_all_subfolders(self, folder = os.getcwd(), folder_list_to_pickle = [], current_folder = []):
        """
        Recursively explore a given folder and extract all subfolders.

        Args:
            folder (str): The path to the folder to explore. Defaults to the current working directory.
            folder_list_to_pickle (list): A list to store the subfolders in the form of a tuple containing the folder's 
                hierarchy and its absolute path. Defaults to an empty list.
            current_folder (list): A list to keep track of the current folder hierarchy during recursive calls. 
                Defaults to an empty list.

        Returns:
            A list of tuples, where each tuple contains the folder's hierarchy and its absolute path.

        Raises:
            None.

        This function recursively explores a given folder and its subfolders to extract their paths and store them in a 
        list of tuples. The first element of the tuple is a string representing the folder hierarchy and the second 
        element is the folder's absolute path. 

        The folder hierarchy is represented as a dot-separated string containing the folder names starting from the root 
        folder. For example, if the root folder is 'my_folder' and it contains two subfolders named 'sub1' and 'sub2', the 
        folder hierarchy for 'sub1' will be 'my_folder.sub1' and the folder hierarchy for 'sub2' will be 'my_folder.sub2'.
        
        This function excludes two specific folders, '__pycache__' and '.ipynb_checkpoints', from the search results.
        """

        logging.debug("%s --> Subfolder analysis", folder)
        # check if all subdirectory of a folder are files or folders (isdir). If dir is folder, append to the folder list and explore  
        for folder_name, folder_path in [(x, os.path.join(folder, x)) for x in os.listdir(folder) if os.path.isdir(os.path.join(folder, x))]:
            if folder_name not in (['__pycache__', '.ipynb_checkpoints']):
                # keep track of the hierarchy of the folder and subfolder
                current_folder.append(folder_name)
                
                # add the current folder to the final list
                folder_list_to_pickle.append((".".join(current_folder), folder_path))

                logging.debug("Added folder %s to the list", folder_name)

                # Recursively explore the current folder for potential subfolders
                folder_list_to_pickle = self.get_all_subfolders(folder_path, folder_list_to_pickle, current_folder)

                # when a folder exploration is finished, we reset the current folder
                current_folder = []
                
        return folder_list_to_pickle
    
        
    def register_all_function_by_value(self):
        """
        This method registers all necessary modules and functions for the proper functioning of the predict_function.
        It searches for all python modules in the current directory and its subdirectories, registers them and their 
        functions to cloudpickle for pickling by value. 
        
        Parameters:
        ----------
        None
        
        Returns:
        -------
        None
        """  
        
        folders_list = [(None, os.getcwd())]
        folders_list.extend(self.get_all_subfolders())

        requirements_set = set()

        # for loop on all files of the project to find all necessary modules and function for the user defined function to save
        for fold_name, folder in folders_list:
            for module in [name for _, name, ispkg in pkgutil.iter_modules([folder]) if ispkg == False]:
                logging.debug("%s --> Analysis", module)

                # If module is in a subfolder, we need to add the subfolder path to the module name
                full_module = module if fold_name is None else fold_name + "." + module

                # extract module spec necessary to pickle
                module_spec = importlib.util.spec_from_file_location(full_module, os.path.join(folder, "{}.py".format(module)))
                module = importlib.util.module_from_spec(module_spec)
                sys.modules[module_spec.name] = module
                module_spec.loader.exec_module(module)

                # and finally, we register the file to pickle
                logging.debug("%s --> Registering", module)
                cloudpickle.register_pickle_by_value(module)

                requirements_set.update(self.find_imported_package(module_spec.origin))

        return requirements_set
            
    def pickle_artifacts(self, artifact_path, predict_function, models):
        """
        Pickles the given objects (predict_function and models) and saves them to the specified artifact_path.
        
        Args:
        - artifact_path (str): Path to the directory where the artifacts will be saved.
        - predict_function (callable): Function associated with the models that needs to be pickled and saved.
        - models (dict): Dictionary of pickable objects (models) associated with the predict_function.
        
        Returns:
        - artifacts (dict): Dictionary containing the name of the artifact (predict function and models) as the key and the path to the saved artifact as the value.
        """


        # Create a dictionary to store artifact names and their corresponding paths
        artifacts = {}

        # Save the predict function
        name_pred = "__predict__"
        path_pred = os.path.join(artifact_path, '{}.pkl'.format(name_pred))
        artifacts[name_pred] = path_pred
        logging.debug("Pickle function --> %s", name_pred)
        logging.debug("Pickle function path --> %s", path_pred)

        with open(path_pred, "wb") as f:
            cloudpickle.dump(predict_function, f)

        logging.debug("Pickle function DONE")

        # Save the models
        for name_model, model in models.items():
            logging.debug("Model --> %s", name_model)
            path_model = os.path.join(artifact_path, "{}.pkl".format(name_model))
            artifacts[name_model] = path_model
            logging.debug("Pickle model --> %s", name_model)
            logging.debug("Pickle model path --> %s", path_model)

            with open(path_model, "wb") as f:
                cloudpickle.dump(model, f)

            logging.debug("Pickle model DONE")
            
        return artifacts
        
                
    def save(self, artifact_path, predict_function, models, run_id):
        """
        Use to save the user defined function (and any associated models)

        Args:
        artifact_path (str): Path in which to save artifacts (function predict and models)
        predict_function (function): Function associated to the models, to save
        models (pickable object): Any pickable object associated with the predict_function

        Returns:
        None

        Saves the user-defined function and its associated models to the given artifact path using MLflow's `log_model` function. 
        The artifacts are pickled using cloudpickle and registered using `register_pickle_by_value` to ensure that all necessary files and functions are included.

        """
        logging.info("Artifact path --> %s",artifact_path)
        
        # Register all necessary files and function usefull to the proper functioning of the predict_function
        requirements_set = self.register_all_function_by_value()
        
        # pickle the given object
        artifacts = self.pickle_artifacts(artifact_path, predict_function, models)
        logging.debug("Artifacts pickled successfully: %s", artifacts)

        # default MLflow function to save a model
        mlflow.pyfunc.log_model(
            artifact_path = "Model",
            python_model  = self,
            artifacts     = artifacts)
        logging.debug("Model saved successfully to MLflow")

        logging.debug("Artifact list --> %s", artifacts)

        artifact_path = "Model"

        req_path = mlflow.artifacts.download_artifacts(run_id=run_id, artifact_path=f"{artifact_path}/requirements.txt")
        
        logged_requirements = self.get_pip_requirements(req_path)
        package_names_logged_req = [re.split('=|<|>', package)[0] for package in logged_requirements]
        requirements_set = [package for package in requirements_set if package not in package_names_logged_req]
        logged_requirements = self.find_imported_package_versions(logged_requirements, requirements_set)

        with open(req_path, 'w') as f:
            f.write("\n".join(logged_requirements))
                
        # remove artifacts from the temporary folder
        for name, _ in artifacts.items():
            os.remove(artifacts[name])
        logging.debug("Artifacts removed from the temporary folder")


    def load_context(self, context):
        """
        Load all the artifacts associated with the MLflow model to load.

        Args:
            context (MLflow object): Default argument of the custom MLflow class. Used to import the artifacts 
                                    loaded by the `load_context` function.

        Returns:
            None

        Raises:
            FileNotFoundError: If any artifact file is missing.

        This function loads all artifacts associated with the MLflow model, including the user-defined 
        predict function and any additional models. It retrieves the default values and arguments of the user-defined 
        predict function and transforms the function to adapt the arguments to the specificity of MLflow. It also 
        saves the artifacts into instance variables of the class for later use.

        """
        
        logging.info("Loading artifacts associated with the MLflow model to load.")
        
        # We need to import those packages inside the function as only the class is pickled with MLflow (not the file).
        # As the loading environment does not necessarily have these packages pre-imported, we import them here.
        import cloudpickle
        import inspect
        
        list_artifacts_name = []
        
        # Loading of all model's associated artifacts.
        for name, artifact in context.artifacts.items():
            list_artifacts_name.append(name)
            logging.debug("Loading artifact %s --> START", name)
            with open(artifact, "rb") as f:
                exec("self.{} =  cloudpickle.load(f)".format(name))
            logging.debug("Loading artifact %s --> DONE", name)

        # Retrieve the specification (args mostly) of the user-defined function.
        signature = inspect.signature(self.__predict__)

        # Retrieve default values of the user-defined function.
        self.arg_default_value = {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
        
        # Retrieve all arguments of the user-defined function.
        list_args_predict = [k for k, v in signature.parameters.items()]

        dict_args___predict__ = {}
        
        # Transform the user-defined function to adapt the arguments of the function to the specificity of MLflow.
        for arg in list_args_predict:
            if arg in list_artifacts_name:
                dict_args___predict__[arg] = "self.{}".format(arg)
            else:
                dict_args___predict__[arg] = "args[\"{}\"]".format(arg)
                
        self.dict_args___predict__ = dict_args___predict__
        
        logging.info("Artifacts loading complete.")

                            
    def predict(self, context, args): 
        """
        Uses the saved function (and any associated models) defined by the user at the time of saving 

        Parameters:
        -----------
        context: `MLflow` object
            Default argument of the custom `MLflow` class. Used to import the artifacts loaded by the `load_context` function.
        args: `dict`
            A dictionary that contains the different arguments used by the predict function.
        
        Returns:
        --------
        The associated return from the user-defined saved function.
        """
        
        # retrieve all default value if no value is given by the user in the predict
        for arg in self.arg_default_value:
            if arg not in args.keys():
                args[arg] = self.arg_default_value[arg]
                logging.debug("Default arg --> %s",arg) 
                logging.debug("Default Value --> %s",self.arg_default_value[arg])
        
        dict_args = {}
        
        for name_arg, value_arg in self.dict_args___predict__.items():
            dict_args[name_arg] = eval(value_arg)  
            logging.debug("Input Arg --> %s",name_arg) 
            logging.debug("Input Value --> %s",eval(value_arg))
            
        logging.debug("START predict")
        result = self.__predict__(**dict_args)
        logging.debug("END predict")
        return result
