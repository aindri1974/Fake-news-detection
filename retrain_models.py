#!/usr/bin/env python
# Model retraining script for the fake news detection system
# This script uses collected feedback data to retrain and improve the models

import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def load_datasets(fake_path=None, true_path=None):
    """Load datasets with fallbacks for different path structures"""
    
    paths_to_try_fake = [
        "Fake.csv/Fake.csv", 
        "../input/fake-news-detection/Fake.csv",
        "Fake.csv"
    ]
    
    paths_to_try_true = [
        "True.csv/True.csv", 
        "../input/fake-news-detection/True.csv",
        "True.csv"
    ]
    
    if fake_path:
        paths_to_try_fake.insert(0, fake_path)
    if true_path:
        paths_to_try_true.insert(0, true_path)
    
    # Try loading from different paths
    df_fake, df_true = None, None
    
    for path in paths_to_try_fake:
        try:
            print(f"Trying to load fake news dataset from: {path}")
            df_fake = pd.read_csv(path)
            print(f"Successfully loaded fake news dataset from: {path}")
            break
        except Exception as e:
            print(f"Failed to load from {path}: {str(e)}")
            continue
            
    for path in paths_to_try_true:
        try:
            print(f"Trying to load true news dataset from: {path}")
            df_true = pd.read_csv(path)
            print(f"Successfully loaded true news dataset from: {path}")
            break
        except Exception as e:
            print(f"Failed to load from {path}: {str(e)}")
            continue
    
    if df_fake is None or df_true is None:
        raise FileNotFoundError("Could not find dataset files")
        
    return df_fake, df_true

def wordopt(text):
    """Text preprocessing function (same as in the notebook)"""
    if not isinstance(text, str):
        return ""
        
    text = text.lower()
    text = re.sub('\\[.*?\\]', '', text)
    text = re.sub("\\\\W"," ",text) 
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)    
    return text

def prepare_data():
    """Prepare and merge the datasets for training"""
    
    print("Loading datasets...")
    try:
        df_fake, df_true = load_datasets()
        
        # Add class labels
        df_fake["class"] = 0  # 0 for fake news
        df_true["class"] = 1  # 1 for real news
        
        # Merge datasets
        df_merge = pd.concat([df_fake, df_true], axis=0)
        
        # Drop unnecessary columns
        df = df_merge.drop(["title", "subject", "date"], axis=1, errors='ignore')
        
        # Check for missing values
        if df.isnull().sum().sum() > 0:
            print(f"Warning: Dataset contains {df.isnull().sum().sum()} missing values")
            df = df.dropna()
        
        # Load user feedback if available
        feedback_file = 'feedback/user_feedback.csv'
        if os.path.exists(feedback_file):
            print("Loading user feedback data...")
            feedback_df = pd.read_csv(feedback_file)
            
            # Process feedback data to match training format
            feedback_processed = pd.DataFrame({
                'text': feedback_df['text'],
                'class': feedback_df['correct_label']
            })
            
            # Add feedback to training data (with higher weight by duplicating)
            # This gives more importance to user-corrected examples
            df = pd.concat([df, feedback_processed, feedback_processed], axis=0)
            print(f"Added {len(feedback_processed)} feedback entries (duplicated for weight)")
        
        # Shuffle data
        df = df.sample(frac=1).reset_index(drop=True)
        
        # Preprocess text
        print("Preprocessing text...")
        df["text"] = df["text"].apply(wordopt)
        
        return df
    
    except Exception as e:
        print(f"Error preparing data: {str(e)}")
        raise

def train_models(df):
    """Train all models on the prepared dataset"""
    
    print("Splitting data into training and testing sets...")
    x = df["text"]
    y = df["class"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
    
    print("Vectorizing text data...")
    vectorizer = TfidfVectorizer()
    xv_train = vectorizer.fit_transform(x_train)
    xv_test = vectorizer.transform(x_test)
    
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Dictionary to hold trained models
    models = {}
    
    # Train Logistic Regression
    print("Training Logistic Regression model...")
    LR = LogisticRegression(max_iter=1000)
    LR.fit(xv_train, y_train)
    pred_lr = LR.predict(xv_test)
    lr_accuracy = accuracy_score(y_test, pred_lr)
    print(f"Logistic Regression Accuracy: {lr_accuracy:.4f}")
    print(classification_report(y_test, pred_lr))
    models['LR'] = LR
    
    # Train Decision Tree
    print("Training Decision Tree model...")
    DT = DecisionTreeClassifier()
    DT.fit(xv_train, y_train)
    pred_dt = DT.predict(xv_test)
    dt_accuracy = accuracy_score(y_test, pred_dt)
    print(f"Decision Tree Accuracy: {dt_accuracy:.4f}")
    print(classification_report(y_test, pred_dt))
    models['DT'] = DT
    
    # Train Gradient Boosting Classifier
    print("Training Gradient Boosting Classifier...")
    GBC = GradientBoostingClassifier(random_state=42)
    GBC.fit(xv_train, y_train)
    pred_gbc = GBC.predict(xv_test)
    gbc_accuracy = accuracy_score(y_test, pred_gbc)
    print(f"Gradient Boosting Classifier Accuracy: {gbc_accuracy:.4f}")
    print(classification_report(y_test, pred_gbc))
    models['GBC'] = GBC
    
    # Train Random Forest Classifier
    print("Training Random Forest Classifier...")
    RFC = RandomForestClassifier(random_state=42)
    RFC.fit(xv_train, y_train)
    pred_rfc = RFC.predict(xv_test)
    rfc_accuracy = accuracy_score(y_test, pred_rfc)
    print(f"Random Forest Classifier Accuracy: {rfc_accuracy:.4f}")
    print(classification_report(y_test, pred_rfc))
    models['RFC'] = RFC
    
    # Save models
    print("Saving models...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save individual models
    joblib.dump(LR, f'model/LR_model.joblib')
    joblib.dump(DT, f'model/DT_model.joblib')
    joblib.dump(GBC, f'model/GBC_model.joblib')
    joblib.dump(RFC, f'model/RFC_model.joblib')
    
    # Also save the vectorizer
    joblib.dump(vectorizer, f'model/tfidf_vectorizer.pkl')
    
    # Backup previous models
    os.makedirs(f'model/backup_{timestamp}', exist_ok=True)
    for model_file in ['LR_model.joblib', 'DT_model.joblib', 'GBC_model.joblib', 'RFC_model.joblib', 'tfidf_vectorizer.pkl', 'fake_news_model.pkl']:
        src_path = f'model/{model_file}'
        if os.path.exists(src_path):
            try:
                import shutil
                shutil.copy(src_path, f'model/backup_{timestamp}/{model_file}')
                print(f"Backup created for {model_file}")
            except Exception as e:
                print(f"Failed to backup {model_file}: {str(e)}")
    
    # Select best model for default model
    accuracies = {
        'LR': lr_accuracy,
        'DT': dt_accuracy,
        'GBC': gbc_accuracy,
        'RFC': rfc_accuracy
    }
    
    best_model_name = max(accuracies, key=accuracies.get)
    best_model = models[best_model_name]
    
    # Save best model as default model
    joblib.dump(best_model, f'model/fake_news_model.pkl')
    print(f"Saved {best_model_name} as the default model (best accuracy: {accuracies[best_model_name]:.4f})")
    
    # Create training report
    report = {
        'timestamp': timestamp,
        'data_size': len(df),
        'accuracies': accuracies,
        'best_model': best_model_name
    }
    
    # Save training report
    os.makedirs('reports', exist_ok=True)
    pd.DataFrame([report]).to_csv(f'reports/training_report_{timestamp}.csv', index=False)
    
    print("Model training complete!")
    return vectorizer, models

if __name__ == "__main__":
    print("Starting model retraining process...")
    try:
        # Prepare and load data
        df = prepare_data()
        print(f"Prepared dataset with {len(df)} entries")
        
        # Train and save models
        vectorizer, models = train_models(df)
        print("Retraining successful!")
        
    except Exception as e:
        import traceback
        print(f"Error during retraining: {str(e)}")
        print(traceback.format_exc())
        exit(1)