**SMS Spam Classification Project**

**Project Overview**

This project aims to classify SMS messages as either spam or non-spam (ham) using various machine learning models. We utilize Natural Language Processing (NLP) techniques to preprocess the text data and extract features using TF-IDF Vectorization. The models evaluated in this project include:

1. Support Vector Machine (SVM) with different kernels (Linear, Polynomial, RBF)
2. Logistic Regression
3. Random Forest Classifier
4. Naive Bayes
5. K-Nearest Neighbors (KNN)

The best-performing model is selected based on accuracy, classification report, and confusion matrix. The trained models and vectorizer are saved for future use.

#### Data Description

The dataset used for this project is `spam.csv`, which contains the following columns:
- `v1`: The label (`ham` for non-spam and `spam` for spam messages).
- `v2`: The SMS message content.

The data undergoes the following preprocessing steps:
- Removal of unnecessary columns.
- Conversion of labels to binary values (`ham` as 0 and `spam` as 1).
- Text cleaning (lowercasing, removing punctuation, numbers, and extra whitespace).

#### Model Training and Evaluation

##### 1. SVM 
Linear Kernel
- **Accuracy**: 98.29%
Polynomial Kernel
**Accuracy**:94.52%
RBF Kernel
**Accuracy**:97.84%

##### 2. Logistic Regression
- **Accuracy**: 96.83%
  
##### 3. Random Forest Classifier
- **Accuracy**: 97.59%
  
##### 4. Naive Bayes
- **Accuracy**: 96.70%
  
##### 5. K-Nearest Neighbors (KNN)
- **Accuracy**: 91.88%

The SVM with a linear kernel achieved the highest accuracy and is selected as the best model.

#### Saving the Model

The SVM with a linear kernel and the TF-IDF vectorizer are saved using `joblib` for future use.

#### Example Usage

An example is provided to demonstrate how to use the saved model and vectorizer to classify new SMS messages.

#### Files Included

- `spam.csv`: The dataset.
- `sms_spam_classification.ipynb`: Jupyter notebook containing the complete code.
- `svm_linear_model.pkl`: Saved SVM model with linear kernel.
- `tfidf_vectorizer.pkl`: Saved TF-IDF vectorizer.
- `README.md`: Project description and instructions.

#### Instructions

1. Clone the repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the Jupyter notebook `sms_spam_classification.ipynb` to see the entire workflow.
4. Use the saved model and vectorizer to classify new SMS messages.

#### Conclusion

This project demonstrates the application of various machine learning models to classify SMS messages. The SVM with a linear kernel was found to be the most effective model for this task, achieving the highest accuracy. The project can be further improved by experimenting with more advanced text processing techniques and additional machine learning models.
