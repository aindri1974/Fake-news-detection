from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from random import randrange
import re
import string
import numpy as np
import os
from datetime import datetime

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Allow cross-origin requests from frontend

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load models if they exist, otherwise use default
try:
    # Try to load all four models used in the notebook
    LR_model = joblib.load("model/LR_model.joblib")
    DT_model = joblib.load("model/DT_model.joblib")
    GBC_model = joblib.load("model/GBC_model.joblib")
    RFC_model = joblib.load("model/RFC_model.joblib")
    vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
    all_models_loaded = True
except FileNotFoundError:
    # Fall back to the single model if not all models are available
    try:
        default_model = joblib.load("model/fake_news_model.pkl")
        vectorizer = joblib.load("model/tfidf_vectorizer.pkl")
        all_models_loaded = False
    except FileNotFoundError:
        raise Exception("No models found. Please ensure model files are in the 'model' directory.")

# Text preprocessing function (same as in the notebook)
def wordopt(text):
    text = text.lower()
    text = re.sub('\\[.*?\\]', '', text)
    text = re.sub("\\\\W"," ",text) 
    text = re.sub('https?://\\S+|www\\.\\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\\n', '', text)
    text = re.sub('\\w*\\d\\w*', '', text)    
    return text

# Input validation function
def validate_text(text):
    """Validate user input"""
    if not isinstance(text, str):
        return False, "Input must be text"
    
    if len(text) < 10:
        return False, "Text is too short for reliable prediction"
        
    if len(text) > 50000:
        return False, "Text is too long (max 50,000 characters)"
    
    return True, None

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # Validate input
        is_valid, error_message = validate_text(text)
        if not is_valid:
            return jsonify({'error': error_message}), 400

        # Preprocess and vectorize
        cleaned_text = wordopt(text)
        features = vectorizer.transform([cleaned_text])
        
        # Check if we have all models or just the default one
        if all_models_loaded:
            # Get predictions from all models
            lr_pred = int(LR_model.predict(features)[0])
            dt_pred = int(DT_model.predict(features)[0])
            gbc_pred = int(GBC_model.predict(features)[0])
            rfc_pred = int(RFC_model.predict(features)[0])
            
            # Get confidence scores if available
            lr_conf = float(LR_model.predict_proba(features)[0][lr_pred]) if hasattr(LR_model, 'predict_proba') else None
            dt_conf = float(DT_model.predict_proba(features)[0][dt_pred]) if hasattr(DT_model, 'predict_proba') else None
            gbc_conf = float(GBC_model.predict_proba(features)[0][gbc_pred]) if hasattr(GBC_model, 'predict_proba') else None
            rfc_conf = float(RFC_model.predict_proba(features)[0][rfc_pred]) if hasattr(RFC_model, 'predict_proba') else None
            
            # Take majority vote for final prediction
            votes = [lr_pred, dt_pred, gbc_pred, rfc_pred]
            final_prediction = max(set(votes), key=votes.count)  # Most common prediction
            
            # Calculate average confidence for the final prediction
            confidences = []
            if lr_pred == final_prediction and lr_conf is not None:
                confidences.append(lr_conf)
            if dt_pred == final_prediction and dt_conf is not None:
                confidences.append(dt_conf)
            if gbc_pred == final_prediction and gbc_conf is not None:
                confidences.append(gbc_conf)
            if rfc_pred == final_prediction and rfc_conf is not None:
                confidences.append(rfc_conf)
                
            avg_confidence = sum(confidences) / len(confidences) if confidences else None
            
            # Build comprehensive result
            result = {
                'label': final_prediction,
                'prediction': 'Fake News' if final_prediction == 0 else 'Real News',
                'confidence': avg_confidence,
                'model_details': {
                    'logistic_regression': {'prediction': lr_pred, 'confidence': lr_conf},
                    'decision_tree': {'prediction': dt_pred, 'confidence': dt_conf},
                    'gradient_boosting': {'prediction': gbc_pred, 'confidence': gbc_conf},
                    'random_forest': {'prediction': rfc_pred, 'confidence': rfc_conf}
                }
            }
        else:
            # Use the default model
            prediction = int(default_model.predict(features)[0])
            
            # Try to get confidence
            confidence = None
            try:
                proba = default_model.predict_proba(features)[0]
                confidence = float(proba[prediction])
            except (AttributeError, NotImplementedError):
                pass
                
            result = {
                'label': prediction,
                'prediction': 'Fake News' if prediction == 0 else 'Real News',
                'confidence': confidence
            }
        
        # Store the prediction in feedback database for future improvements
        try:
            store_prediction(text, result)
        except:
            # Don't fail if storage fails
            pass
            
        return jsonify(result)
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/feature_importance', methods=['POST'])
def feature_importance():
    """Endpoint that returns the most important features (words) for a prediction"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        
        # Validate input
        is_valid, error_message = validate_text(text)
        if not is_valid:
            return jsonify({'error': error_message}), 400
            
        # Preprocess text
        cleaned_text = wordopt(text)
        features = vectorizer.transform([cleaned_text])
        
        # Get the feature names (words) from the vectorizer
        feature_names = vectorizer.get_feature_names_out()
        
        # Get important features based on model type
        important_features = []
        
        if all_models_loaded:
            # Logistic Regression coefficients are the easiest to interpret
            if hasattr(LR_model, 'coef_'):
                # Get prediction to determine which class's features to highlight
                prediction = int(LR_model.predict(features)[0])
                coef = LR_model.coef_[0]  # For binary classification
                
                # Get the feature indices that are present in the text
                feature_indices = features[0].nonzero()[1]
                
                # Create a list of (word, importance) tuples
                feature_importance = []
                for idx in feature_indices:
                    word = feature_names[idx]
                    importance = coef[idx]
                    # If predicting fake news (0), negative coefficients are more important
                    # If predicting real news (1), positive coefficients are more important
                    if (prediction == 0 and importance < 0) or (prediction == 1 and importance > 0):
                        feature_importance.append((word, abs(importance), "supporting"))
                    else:
                        feature_importance.append((word, abs(importance), "contradicting"))
                
                # Sort by absolute importance
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                # Take top 20 features
                for word, importance, contribution in feature_importance[:20]:
                    important_features.append({
                        'word': word,
                        'importance': float(importance),
                        'contribution': contribution
                    })
        elif hasattr(default_model, 'coef_'):
            # Similar logic for default model if it has coefficients
            prediction = int(default_model.predict(features)[0])
            coef = default_model.coef_[0]
            feature_indices = features[0].nonzero()[1]
            
            feature_importance = []
            for idx in feature_indices:
                word = feature_names[idx]
                importance = coef[idx]
                if (prediction == 0 and importance < 0) or (prediction == 1 and importance > 0):
                    feature_importance.append((word, abs(importance), "supporting"))
                else:
                    feature_importance.append((word, abs(importance), "contradicting"))
            
            feature_importance.sort(key=lambda x: x[1], reverse=True)
            
            for word, importance, contribution in feature_importance[:20]:
                important_features.append({
                    'word': word,
                    'importance': float(importance),
                    'contribution': contribution
                })
        
        return jsonify({
            'important_features': important_features,
            'prediction': 'Fake News' if prediction == 0 else 'Real News'
        })
    except Exception as e:
        import traceback
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    """Endpoint to collect user feedback on predictions"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        predicted_label = data.get('predicted_label')
        correct_label = data.get('correct_label')
        
        if not all([isinstance(predicted_label, int), isinstance(correct_label, int)]):
            return jsonify({'error': 'Labels must be integers (0 for fake, 1 for real)'}), 400
            
        feedback_data = {
            'text': text,
            'predicted_label': predicted_label,
            'correct_label': correct_label,
            'timestamp': datetime.now().isoformat()
        }
        
        # Create feedback directory if it doesn't exist
        os.makedirs('feedback', exist_ok=True)
        
        # Append to feedback CSV
        feedback_df = pd.DataFrame([feedback_data])
        feedback_file = 'feedback/user_feedback.csv'
        
        if os.path.exists(feedback_file):
            feedback_df.to_csv(feedback_file, mode='a', header=False, index=False)
        else:
            feedback_df.to_csv(feedback_file, index=False)
            
        return jsonify({'status': 'success', 'message': 'Feedback recorded'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def store_prediction(text, result):
    """Store prediction results for future analysis and model improvement"""
    try:
        # Create predictions directory if it doesn't exist
        os.makedirs('predictions', exist_ok=True)
        
        # Prepare prediction data
        prediction_data = {
            'text': text,
            'label': result.get('label'),
            'confidence': result.get('confidence'),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add individual model predictions if available
        if 'model_details' in result:
            for model_name, details in result['model_details'].items():
                prediction_data[f'{model_name}_prediction'] = details.get('prediction')
                prediction_data[f'{model_name}_confidence'] = details.get('confidence')
        
        # Append to predictions CSV
        pred_df = pd.DataFrame([prediction_data])
        pred_file = 'predictions/stored_predictions.csv'
        
        if os.path.exists(pred_file):
            pred_df.to_csv(pred_file, mode='a', header=False, index=False)
        else:
            pred_df.to_csv(pred_file, index=False)
    except Exception as e:
        # Log error but don't crash the app
        print(f"Error storing prediction: {e}")
        pass

@app.route('/random', methods=['GET'])
def random():
    try:
        data = pd.read_csv("random_dataset.csv")
        index = randrange(0, len(data)-1, 1)
        return jsonify({
            'text': data.loc[index].text,
            'label': int(data.loc[index].label) if 'label' in data.columns else 0
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint to verify the API is working properly"""
    models_status = "All models loaded" if all_models_loaded else "Using default model only"
    return jsonify({
        'status': 'ok', 
        'message': 'Fake News Detection API is operational',
        'models': models_status
    })

@app.route('/models', methods=['GET'])
def get_models():
    """Endpoint to check which models are loaded"""
    if all_models_loaded:
        return jsonify({
            'status': 'ok',
            'models': ['logistic_regression', 'decision_tree', 'gradient_boosting', 'random_forest']
        })
    else:
        return jsonify({
            'status': 'partial',
            'models': ['default_model']
        })

if __name__ == '__main__':
    app.run(debug=True)