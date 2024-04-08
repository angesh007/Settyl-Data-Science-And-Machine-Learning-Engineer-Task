   step 1-> Data Preprocessing:
        Run the data_preprocessing.py script to preprocess the dataset.
        Ensure that the preprocessing script handles loading the dataset, cleaning data, encoding categorical variables, and splitting it into training and validation sets.
        -> python data_preprocessing.py(in integrated terminal)
 
   Step 2->Model Development:
    Run the model_development.py script to develop and train the machine learning model.
    This script should load the preprocessed dataset, define the model architecture, train the model, and evaluate its performance
     ->python model_development.py(in integrated terminal)

   step 3->API Implementation:
        first use ->>pip install fastapi uvicorn numpy tensorflow scikit-learn(in integrated terminal)
        Run the FastAPI application using Uvicorn.
        Ensure that the api.py script defines the FastAPI application and includes the endpoints for prediction and testing.
       -> uvicorn api:app --reload(in integrated terminal)
    
  You can use tools like cURL or Postman to send HTTP requests to the API endpoints /predict/ and /test/ and /validate the responses
         http://127.0.0.1:8000/predict/
         http://127.0.0.1:8000/test/
