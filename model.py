import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import pickle

def process(msg):
    # Load or preprocess data
    raw_mail_data = pd.read_csv('./spam.csv')
    mail_data = raw_mail_data.where(pd.notnull(raw_mail_data), '')

    # Convert category labels to numerical values
    mail_data['Category'] = mail_data['Category'].map({'spam': 0, 'ham': 1})

    # Split data into features (X) and labels (Y)
    X = mail_data['Message']
    Y = mail_data['Category']

    # Split data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=3)

    # Extract features from text data
    feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)
    X_train_features = feature_extraction.fit_transform(X_train)
    X_test_features = feature_extraction.transform(X_test)

    # Convert labels to integers
    Y_train = Y_train.astype('int')
    Y_test = Y_test.astype('int')

    # Define model file path
    model_file = 'spam_classifier_model.pkl'

    # Check if model file exists
    if os.path.exists(model_file):
        # Load the trained model
        with open(model_file, 'rb') as file:
            model = pickle.load(file)
    else:
        # Train the model if model file doesn't exist
        model = LogisticRegression()
        model.fit(X_train_features, Y_train)
        
        # Save the trained model
        with open(model_file, 'wb') as file:
            pickle.dump(model, file)

    # Evaluate the model on training and test data
    train_accuracy = accuracy_score(Y_train, model.predict(X_train_features))
    test_accuracy = accuracy_score(Y_test, model.predict(X_test_features))
    print(f'Training accuracy: {train_accuracy:.2f}')
    print(f'Test accuracy: {test_accuracy:.2f}')

    # Example of classifying an email
    input_mail = [msg]
    input_data_features = feature_extraction.transform(input_mail)
    prediction = model.predict(input_data_features)

    if prediction[0] == 1:
        return 'Ham message'
    else:
        return 'Spam message'
