
![alt text](https://github.com/SimonLMC/cloudflow/blob/main/image/cloudflow_logo.svg?raw=true)

# 💡 What is __Cloudflow__ ? 

__Cloudflow is a Python package that combines two powerful tools for machine learning model management: Cloudpickle and MLflow.__

- Cloudpickle is a serialization library that allows Python objects to be serialized and deserialized across different Python processes or even different Python versions. This is especially useful for machine learning models, which can be complex and difficult to share between different environments.

- MLflow, on the other hand, is an open source platform for the complete machine learning lifecycle that helps to manage experiments, package code into reproducible runs, and share and deploy models. 

By integrating Cloudpickle with MLflow, __CloudFlow__ provides an easy and efficient way to serialize and save complex machine learning models, track your experiments and their results, and share your models with others in a reproducible and scalable way.

Whether you are working on a single project or collaborating with a team, Cloudflow can help you manage your machine learning models more effectively and streamline your machine learning workflow.

- One of the key benefits of Cloudflow is that it makes it easy to __save and manage complex combinations models__. 
With traditional approaches, it can be challenging to __combine multiple models into a single object that can be easily serialized and saved__. However, with Cloudflow, you can easily combine multiple models, __along with their associated pre-processing and post-processing steps__, into a single model object that can be easily serialized and saved. This makes it simple to manage complex combinations of models, which can be very useful when you are working on more advanced machine learning problems.

- Another powerful feature of Cloudflow is its ability to __hide your code__. When you use Cloudflow to serialize a machine learning models and associated functions, you exclude the original Python code as the resulting object is serialized. This can be particularly useful when you want to share a model but you don't want to reveal the underlying code. By hiding the code, you can protect your intellectual property and ensure that your models remain secure. This feature is especially important when working with sensitive data, where the privacy and security of the models is paramount.

- [Installation](#installation)
- [Getting started](#getting-started)
- [API](#api)

## 🦾 Installation

> You need Python 3.6 or above.
From the terminal (or Anaconda prompt in Windows), enter:

```bash
pip install cloudflow
```

## 🚀 Getting started

- [Saving](#Saving)
- [Loading](#Loading)

## 💾 Saving

__All code from the saving part of this section can be found in demo/MLFLOW_save folder__

First, let's train/import some models.
To simplify the demonstration, we will use models from the HuggingFace library. However, any other type of model can be used.
By using pre-trained models from HuggingFace, the demo code can focus on demonstrating the functionality of the cloudflow package, rather than on the complexities of model training and selection. This approach also makes it easier to reproduce the demo code, since the pre-trained models can be easily accessed and used by anyone who has access to the HuggingFace library.

```python
from transformers import pipeline

summarization_model  = pipeline("summarization")        # Summarize a text
sentiment_model      = pipeline("sentiment-analysis")   # Sentiment analysis
translation_model    = pipeline('translation_en_to_fr') # Traduction from english to french
image_classification = pipeline("image-classification") # Image Classification

```
This block of code imports the required packages pipeline from the Huggingfaces library. 
It also imports four pipelines for different NLP/Image classifcation tasks - summarization, sentiment analysis, 
translation, and image classification - are initialized using the pipeline() method.


Then, lets create a function that will combine all those models.
We will also use differents functions from differents modules to simulate a complexe project.

```python
import download_file

def predict_to_save(sentiment_model, summarization_model, translation_model, image_classification, input_type, data=None, min_length=0, max_length=150):
    """
    This function combines different machine learning models for sentiment analysis, summarization, translation, and image classification.

    Parameters:
    -----------
    sentiment_model : object
        The pipeline object for sentiment analysis.
    summarization_model : object
        The pipeline object for summarization.
    translation_model : object
        The pipeline object for translation.
    image_classification : object
        The pipeline object for image classification.
    input_type : str
        The type of input data, either "sentiment", "translation", "image", or "summarization".
    data : str, optional
        The input data to be used for prediction. Required only for "summarization" input type, and optional for "image".
    min_length : int, optional
        The minimum length of the summary (used only for "summarization" input type).
    max_length : int, optional
        The maximum length of the summary (used only for "summarization" input type).

    Returns:
    --------
    result : object
        The predicted result for the given input data and type.

    Raises:
    -------
    ValueError
        If an invalid input type is selected.
    Exception
        If there is an error during prediction.

    Examples:
    ---------
    >>> predict_to_save(sentiment_model, summarization_model, translation_model, image_classification, input_type = "sentiment", "I am happy today.")
    "Positive"
    >>> predict_to_save(sentiment_model, summarization_model, translation_model, image_classification, input_type = "translation", "Hello, how are you?")
    "Bonjour, comment allez-vous?"
    >>> predict_to_save(sentiment_model, summarization_model, translation_model, image_classification, input_type = "image", "https://example.com/image.jpg")
    {"category": "dog", "confidence": 0.95}
    >>> predict_to_save(sentiment_model, summarization_model, translation_model, image_classification, input_type = "summarization", "https://example.com/story.txt")
    {"summary": "A summary of the story.", "input_text": "The full text of the story."}
    """
    
    if input_type not in ["sentiment", "translation", "image", "summarization"]:
        raise ValueError("Invalid input type selected")

    try:
        if input_type == "sentiment":
            return sentiment_model(data)
        elif input_type == "translation":
            return translation_model(data)
        elif input_type == "image":
            image = download_file.download_image(data)
            return image_classification(image)
        elif input_type == "summarization":
            if data is None:
                data = download_file.download_story()
            summary = summarization_model(data, min_length=min_length, max_length=max_length)[0]
            result = {"summary": summary, "input_text": data}
            return result
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")

```

This code defines a function called predict_to_save, which select model to predict with from different types of pre-trained models based on the input data type. The function takes in five required parameters, sentiment_model, summarization_model, translation_model, image_classification, and input_type, as well as three optional parameters, data, min_length, and max_length.

The purpose of the function is to make predictions using one or more of the provided pre-trained models based on the input_type. 
- If input_type is "sentiment", sentiment_model will be used to predict the sentiment of the input data. 
- If input_type is "translation", translation_model will be used to translate the input data. 
- If input_type is "image", image_classification will be used to classify the input image. 
- If input_type is "summarization", summarization_model will be used to summarize the input text.
- If the input type is "summarization" and no data is provided, the function uses a story downloaded by the download_story function defined in the download_file module. 

The demo code built to showcase the capabilities of Cloudflow to work with complex project structures, such as the one shown below:


    ┌── sub_folder/             # Sub-folder
        ├── sub_sub_folder/     # Sub-folder within the sub-folder
            ├── is_sub.py       # Python file for printing a message to the console 
    ├── download_file.py        # Python file for downloading Image for image classif
    ├── get_print.py            # Python file for printing a message to the console
    ├── intermediate_module.py  # Python file for an intermediate module that connects different modules
    ├── utils.py                # Python file for utility functions
    └── SAVE_MLflow.ipynb       # Main notebook file for demonstration


As you can see, the project folder has multiple files and folders, including subfolders with further subfolders. This project structure is more complex than a basic Python script, but the Cloudflow package can still be used to manage the workflow. The demo code showcases how to integrate the Cloudflow package with this type of project structure and provides examples of how to use the package to manage the workflow, track the results, and save the models.


finaly, we save our project:

```python
#import the necessary packages.
import mlflow
import cloudflow

#specify the folder path where to save the model and the name of the experiment.
tracking_uri ="/saving_path"
experiment_id = "experiment_name" 

# prepare the environment for logging the model and the results.
cloudflow.prepare_env(tracking_uri,experiment_id)

#start an MLflow run for the specified experiment and assign it to the run object.
with mlflow.start_run(experiment_id = experiment_id) as run:
    
    print("RUN ID : ", run.info.run_id)
    
    # log a metric in the current MLflow run, with the name 'test_metrics' and the value 0.99.
    mlflow.log_metric('test_metrics', 0.99)

    # create a cloudflow_model object with the specified debugging level ("INFO", "DEBUG", "ERROR", "WARNING").
    model = cloudflow.cloudflow_model()

    # save the cloudflow_model object to the specified file path and experiment, with the specified run_id. 
    # The predict_function is the function to use for making predictions, 
    # and the models dictionary contains the models to include in the saved model.        
    model.save(tracking_uri    = tracking_uri,
               experiment_id = experiment_id,
               run_id = run.info.run_id, 
               predict_function = predict_to_save, 
               models = {"summarization_model"  : summarization_model, 
                         "image_classification" : image_classification,
                         "translation_model"    : translation_model,
                         "sentiment_model"      : sentiment_model})
```

The code demonstrates just how easy it is to use the cloudflow packages to log and save machine learning models. By importing the necessary packages, specifying the file path and experiment name, and preparing the environment for logging, this script quickly sets up a system for tracking the results of a machine learning experiment. With just a few lines of code, a metric can be logged in the current MLflow run, and a cloudflow_model object can be created and saved to the specified file path and experiment.

## 😎 Loading

__All code from the Loading part of this section can be found in demo/MLFLOW_load folder__

Now, let's load our project:

A saved model can be loaded in just a few lines of code. 

The load_model() method of the cloudflow_model object is used to load the saved model from the specified tracking URI, experiment ID, and run ID. Once the model is loaded, it can be used for making predictions by calling its predict function. 

```python
import cloudflow

loaded_model = cloudflow.cloudflow_model()

loaded_model.load_model(tracking_uri  = "/saving_path"
                        experiment_id = "experiment_name" ,
                        run_id        = "# your run id") #from the run.info.run_id

```

Finaly, we can use the saved predict_function

```python
loaded_model.predict(input_type = "image",
                     data       = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/coco_sample.png")

```
more examples can be found in the MLFLOW_load notebook.


## 📖 API

You can choose your logging level when you initialize the cloudflow model
```python
cloudflow_model = cloudflow.cloudflow_model(log_level = "DEBUG") #("INFO", "DEBUG", "ERROR", "WARNING")
```


### cloudflow_model.save(tracking_uri, experiment_id, run_id, predict_function, models)

Save a predict function as an cloudflow model using cloudpickle.
#### Parameters

- __tracking_uri__ (str): The path to the folder where the MLflow experiment will be saved.
- __experiment_id__ (str): The name of the MLflow experiment in which the model will be saved.
- __run_id__ (str): The ID of the run (model ID).
- __predict_function__ (callable): The user-defined predict function to save.
- __models__ (dict): A dictionary of all models used in the predict function. The dictionary should have the following format:
```python
    { "model_name_1": model_1,
      "model_name_2": model_2,
    ... }
```
where each key is a string representing the name of the model (it must be the exact name used in the predict function), and each value is an object representing the corresponding model.



### cloudflow_model.load_model(tracking_uri,experiment_id ,run_id)

Load a desired Cloudflow model.

#### Parameters

- __tracking_uri__ (str): The path to the folder where the MLflow experiment is saved.
- __experiment_id__ (str): The name of the MLflow experiment where the model is saved.
- __run_id__ (str): The ID of the run (model ID) to load.
#### Returns
- __MLflow_model__(callable): The loaded MLflow model.


### cloudflow_model.predict(self, **params)

Execute the user-defined saved predict function of the loaded model.

#### Parameters

- __params__ :
    - (Any): params of the saved predict_function
#### Returns
- (Any): The expected output of the saved predict function.


### prepare_env(tracking_uri, experiment_id)

This function prepares the environment for an MLflow experiment by setting the tracking URI and creating a new experiment with the specified experiment ID if it doesn't already exist.

#### Parameters
- __tracking_uri__ (str): The URI for tracking the MLflow experiment.
- __experiment_id__ (str): The ID of the experiment to create or use.
#### Returns
- None



