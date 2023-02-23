
![alt text](https://github.com/SimonLMC/cloudflow/blob/main/image/cloudflow_logo.svg?raw=true)

# 💡 What is __Cloudflow__ ? 

Cloudflow is a Python package that combines two powerful tools for machine learning model management: Cloudpickle and MLflow. 

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
To simplify the demonstration and its reproducibility, we will use models from HuggingFace. However, any other type of model can be used.

```python
from transformers import pipeline

summarization_model  = pipeline("summarization")        # Summarize a text
sentiment_model      = pipeline("sentiment-analysis")   # Sentiment analysis
translation_model    = pipeline('translation_en_to_fr') # Traduction from english to french
image_classification = pipeline("image-classification") # Image Classification

```
Then, lets create a function that will combine all those models.
We will also use differents functions from differents modules to simulate a complexe project.
The structure of the projet is as follow.

    .
    ├── ...
    ├── sub_folder                 # 
    │   ├── sub_sub_folder         # 
    │       ├── is_sub.py          # 
    ├── download_file.py           # 
    ├── get_print.py               # 
    ├── intermediate_module.py     # 
    ├── utils.py                   # 
    ├── SAVE_MLflow.py             #
    └── ...


```python
import download_file

def predict_to_save(sentiment_model,summarization_model, translation_model,image_classification,input_type, data = None,min_length = 0, max_length = 150):
    
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

        dict_result = summarization_model(data, min_length, max_length)[0]
        dict_result["input_text"] = data
        return dict_result
```

Then, we save our project:

```python
import cloudflow

tracking_uri  = "/saving_path"
experiment_id = "experiment_name" 

cloudflow.prepare_env(tracking_uri,experiment_id)

with mlflow.start_run(experiment_id = experiment_id) as run:
    
    print("RUN ID : ", run.info.run_id)

    mlflow.log_metric('test_metrics', 0.99)

    model = cloudflow.cloudflow_model()        
    model.save(tracking_uri     = tracking_uri,
               experiment_id    = experiment_id,
               run_id           = run.info.run_id, 
               predict_function = predict_to_save, 
               models           = {"summarization_model"  : summarization_model, 
                                    "image_classification" : image_classification,
                                    "translation_model"    : translation_model,
                                    "sentiment_model"      : sentiment_model})

```

## 😎 Loading

__All code from the Loading part of this section can be found in demo/MLFLOW_load folder__

Now, let's load our project:

```python
import cloudflow

loaded_model = cloudflow.cloudflow_model()

loaded_model.load_model(tracking_uri  = "/saving_path"
                        experiment_id = "experiment_name" ,
                        run_id        = "# your run id")

```

And Finaly, we can use the saved predict_function

```python
loaded_model.predict(input_type = "image",
                     data       = "https://huggingface.co/  datasets/huggingface/documentation-images/resolve/main/coco_sample.png")

```

## 📖 API

You can choose your logging level when you initialize the cloudflow model
```python

cloudflow_model = cloudflow.cloudflow_model(log_level = "DEBUG")

```


### cloudflow_model.save(tracking_uri, experiment_id, run_id, predict_function, models)

Save a predict function as an mlflow model using cloudpickle.
#### Parameters

- __tracking_uri__ : str
    - path to folder in which save your differents MLflow experiment
- __experiment_id__ : str
    - Name of the experiment in which save you model
- __run_id__ : str
    - ID of the run (model ID)
- __predict_function__ : function
    - Defined predict function to save
- __models__ : dict
    - Dictionnary of all models used in the predict_function. 
        - Key str : model_name (note: it must be the exact name used in the predict_function)
        - Value object : model



### cloudflow_model.load_model(tracking_uri,experiment_id ,run_id)

Load the desired model
#### Parameters

- __tracking_uri__ : str
    - Path to folder in which your differents MLflow experiment to load
- __experiment_id__ : str
    - Name of the experiment in which you saved the model to load
- __run_id__ : str
    - ID of the run (model ID) to load
#### Returns

- __MLflow_model__:
    - MLflow model


### cloudflow_model.predict(self, **params)

execute the user defined saved predict_function of the loaded model.
#### Parameters

- __params__ :
    - params of the saved predict_function
#### Returns
- the expected return of the saved predict_function


