# Churn-Predictions
PROJECT TITLE
-------------
Customer Churn Prediction using Artificial Neural Network (ANN)


PROJECT DESCRIPTION
-------------------
This project is focused on predicting whether a customer will stay with
the company or leave (churn) using an Artificial Neural Network (ANN).

Customer churn prediction is an important problem in the banking and
service industry, as retaining existing customers is more cost-effective
than acquiring new ones.

The model is trained on a churn dataset containing customer demographic
and account-related information. After training, the model predicts the
probability of churn for new customers and is deployed using Streamlit
for easy interaction.


PROJECT OBJECTIVE
-----------------
- To analyze customer data
- To build and train an ANN model for churn prediction
- To predict whether a customer will stay or leave the company
- To deploy the trained model using Streamlit


PROJECT WORKFLOW
----------------
The project is divided into three main parts:

1. Model Training
   - Data loading and preprocessing
   - Feature scaling and encoding
   - Building an Artificial Neural Network
   - Training and evaluating the model
   - Saving the trained model and preprocessor

2. Model Prediction
   - Loading the saved ANN model
   - Preprocessing new customer input data
   - Predicting churn probability

3. Deployment using Streamlit
   - Creating a user-friendly web interface
   - Taking customer details as input
   - Displaying churn prediction results in real time


DATASET
-------
The dataset contains customer-related features such as:
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Member Status

Target Variable:
- Exited (0 = Customer stays, 1 = Customer leaves)


MODEL DETAILS
-------------
- Model Type: Artificial Neural Network (ANN)
- Output: Binary Classification (Churn / No Churn)
- Activation Function:
  - Hidden Layers: ReLU
  - Output Layer: Sigmoid
- Loss Function: Binary Crossentropy
- Optimizer: Adam
- Evaluation Metric: Accuracy


TECH STACK
----------
Programming Language:
- Python

Libraries and Frameworks:
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Streamlit
- Pickle

Tools:
- Jupyter Notebook
- VS Code / Anaconda
- Streamlit Web App


DEPLOYMENT
----------
The trained ANN model is deployed using Streamlit.
Users can enter customer details through the web interface,
and the application predicts whether the customer is likely
to churn or stay with the company.


HOW TO RUN THE PROJECT
----------------------
1. Install required libraries:
   pip install numpy pandas scikit-learn tensorflow streamlit

2. Run the training notebook to generate:
   - model.h5
   - preprocessor.pkl

3. Start the Streamlit application:
   streamlit run app.py

4. Open the browser and interact with the app.


RESULT
------
The model successfully predicts customer churn based on
input features and provides a probability score indicating
the likelihood of the customer leaving the company.


AUTHOR
------
Name: Ayush Kumar Bhattacharjee
Project Type: Machine Learning / Deep Learning

Email: ayushbhattacharya111@gmail.com
LinkediN: https://linkedin.com/in/ayush-kumar-bhattacharjee-2b2737324

ACKNOWLEDGEMENT
---------------
This project is developed for learning and academic purposes,
demonstrating the use of Artificial Neural Networks and
deployment of machine learning models using Streamlit.
