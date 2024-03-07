# Illinois Data Science Club Spring 2023 Team CWMDSJ Project

## Introduction

Heart disease is a significant health concern globally, with increasing prevalence and associated mortality rates. According to the World Health Organization, cardiovascular diseases (CVDs) are the number one cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Our development of a classification prediction model for heart disease can be a game-changer. Such a model can help predict the likelihood of heart disease based on various factors such as age, gender, and medical history. This predictive model can be instrumental in early detection, prevention, and management of heart disease, thereby potentially saving millions of lives each year.

## Purpose

**Prevent Future Heart Diesase:**
 By analyzing data from past heart disease cases, we can identify patterns and trends that may have contributed to the disease's onset. This information can be used to develop new preventative measures and procedures that can prevent similar cases in the future.

 **Improving Treatment:**
  Investigating data about heart disease can also help identify issues with current treatment methods or highlight the need for personalized treatment plans. This information can be used to make improvements in these areas to increase the effectiveness of treatments and improve patient outcomes.

**Insurance Purposes:**
Additionally, heart disease data can be used in legal proceedings to determine liability and damages in cases of medical malpractice or insurance claims. This information is important for insurance companies, lawyers, and courts to make informed decisions about who is responsible and what compensation may be owed to the victims.

## Kaggle Dataset

Our data is based off a Kaggle available data set, the Cleveland Clinic Heart Disease Dataset.

<img width="616" alt="image" src="https://github.com/UIUC-DSC/CWMDSJ/assets/132399910/13aab53d-59a9-4a21-b6f7-00ba17e4af77">

## Column Descriptions

 - Age - age in years
 - Sex - (1 = male; 0 = female)
 - CP - chest pain type
 - Trestbps - resting blood pressure (in mm Hg on admission to the hospital)
 - Chol - serum cholesterol in mg/dl
 - FBS - (fasting blood sugar &gt; 120 mg/dl) (1 = true; 0 = false)
 - Restecg - resting electrocardiographic results
 - Thalach - maximum heart rate achieved
 - Exang - exercise induced angina (1 = yes; 0 = no)
 - Oldpeak - ST depression induced by exercise relative to rest
 - Slope - the slope of the peak exercise ST segment
 - Ca - number of major vessels (0-3) colored by fluoroscopy
 - Thal - 1 = normal; 2 = fixed defect; 3 = reversible defect
 - Num - artery diameter (0-4)

## Visual Description

**Question:** Which categories show the highest correlation with the classification
**Response:** Visualize each category via a box plot

<img width="552" alt="image" src="https://github.com/UIUC-DSC/CWMDSJ/assets/132399910/54781d1a-8526-4f9d-9b7f-f2d02127028f">

 - num on the y axis is the target variable, showing severity of heart disease (0 being none, 4 being fatal)

The plots above show clear correlations. Other plots didn't. We removed a few that showed little correlation if any. Due to these changes made, the accuracy of our model jumped up by around 10%. 

Another tool we used was the correlation matrix/heat map.

<img width="644" alt="image" src="https://github.com/UIUC-DSC/CWMDSJ/assets/132399910/e1fa3155-6821-46ee-b486-7ad52ff8069d">

## Model Analysis

**Purpose:**
We are aiming to predict the occurrence of heart disease using various health parameters. We have employed a Multilayer Perceptron (MLP) Classifier, a type of neural network, to create a predictive model. The model is trained on a subset of the data and then tested on an unseen dataset to evaluate its performance. The accuracy of the model is calculated as a measure of its predictive power. Additionally, a correlation matrix is generated to understand the relationships between different features and the target variable. The ultimate goal of this model is to aid in the early detection and prevention of heart disease, thereby potentially improving patient outcomes and saving lives.

**Model Specifics:**
The machine learning model we're using is a Multi-Layer Perceptron (MLP), which is a type of artificial neural. The MLP we use has two hidden layers, with 32 neurons in the first layer and 16 neurons in the second layer. The activation function for the neurons is ReLU (Rectified Linear Unit), and the optimizer used for training the network is Adam. The model determines the weights of each layer based on previous iterations of epoch training. 

MLP is a good choice for this classification task because it can model complex, non-linear relationships between the features and the target variable. 

To improve the model's performance in the future, we could consider the following:
1. Hyperparameter Tuning: You could experiment with different values for the hyperparameters of the MLP, such as the number of hidden layers, the number of neurons in each layer, the activation function, and the learning rate of the optimizer.
2. Feature Engineering: You could create new features that might be relevant for predicting heart disease, or use techniques like PCA (Principal Component Analysis) to reduce the dimensionality of your dataset.
3. Different Models: You could try other types of models, such as Support Vector Machines (SVM), Random Forests, or Gradient Boosting Machines (GBM), and compare their performance with the MLP.
4. Ensemble Methods: You could combine the predictions of several models to improve the accuracy of the final


## Results
We were able to implement a model that achieved a 95% accuracy rate in heart disease prediction. 
