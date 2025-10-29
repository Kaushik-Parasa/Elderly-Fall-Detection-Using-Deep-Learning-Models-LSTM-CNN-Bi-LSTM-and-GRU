#!/usr/bin/env python3
"""
Test script to load and evaluate existing trained models
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def generate_test_data(n_samples=200, sequence_length=600, n_features=6):
    """Generate test data for model evaluation"""
    print("Generating test data...")
    
    # Generate synthetic fall data (class 1)
    fall_data = []
    for i in range(n_samples // 2):
        data = np.random.randn(sequence_length, n_features) * 0.5
        # Add fall signature
        spike_start = np.random.randint(100, 400)
        data[spike_start:spike_start+50, :3] += np.random.randn(50, 3) * 3
        fall_data.append(data)
    
    # Generate synthetic ADL data (class 0)
    adl_data = []
    for i in range(n_samples // 2):
        data = np.random.randn(sequence_length, n_features) * 0.8
        adl_data.append(data)
    
    X = np.array(fall_data + adl_data)
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))
    
    # Normalize data
    for i in range(X.shape[2]):
        min_val = np.min(X[:, :, i])
        max_val = np.max(X[:, :, i])
        X[:, :, i] = 2 * (X[:, :, i] - min_val) / (max_val - min_val) - 1
    
    return X, y

def test_model(model_path, X_test, y_test, model_name):
    """Test a single model"""
    try:
        print(f"\nTesting {model_name}...")
        print(f"Loading model from: {model_path}")
        
        # Load model
        model = keras.models.load_model(model_path)
        print(f"Model loaded successfully!")
        print(f"Model input shape: {model.input_shape}")
        
        # Make predictions based on model type
        if isinstance(model.input_shape, list):
            # Multi-input model (Bi-LSTM+CNN)
            predictions = model.predict([X_test[:, :, :3], X_test[:, :, 3:6]])
        else:
            # Single input model
            predictions = model.predict(X_test)
        
        # Convert to binary predictions
        binary_predictions = (predictions > 0.5).astype(int)
        
        # Calculate accuracy
        accuracy = np.mean(binary_predictions.flatten() == y_test)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, binary_predictions)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'plots/{model_name}_test_confusion_matrix.png')
        plt.show()
        
        # Classification Report
        print(f"\nClassification Report for {model_name}:")
        print(classification_report(y_test, binary_predictions))
        
        return accuracy
        
    except Exception as e:
        print(f"Error testing {model_name}: {str(e)}")
        return None

def main():
    """Main function"""
    print("Testing Existing Fall Detection Models")
    print("=" * 50)
    
    # Create plots directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Generate test data
    X_test, y_test = generate_test_data(n_samples=400)
    print(f"Test data shape: X={X_test.shape}, y={y_test.shape}")
    
    # Find all model files
    model_dir = 'models'
    if not os.path.exists(model_dir):
        print(f"Models directory '{model_dir}' not found!")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    
    if not model_files:
        print("No .h5 model files found in the models directory!")
        return
    
    print(f"Found {len(model_files)} model files:")
    for f in model_files:
        print(f"  - {f}")
    
    # Test each model
    results = {}
    
    for model_file in model_files:
        model_path = os.path.join(model_dir, model_file)
        model_name = model_file.replace('.h5', '').replace('_trained_model', '')
        
        accuracy = test_model(model_path, X_test, y_test, model_name)
        if accuracy is not None:
            results[model_name] = accuracy
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    if results:
        for model_name, accuracy in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{model_name:30} - Accuracy: {accuracy:.4f}")
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        models = list(results.keys())
        accuracies = list(results.values())
        
        plt.bar(models, accuracies, color='skyblue', alpha=0.7)
        plt.title('Model Performance Comparison')
        plt.xlabel('Model')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for i, v in enumerate(accuracies):
            plt.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('plots/model_comparison.png')
        plt.show()
        
        print(f"\nBest performing model: {max(results, key=results.get)} ({max(results.values()):.4f})")
    else:
        print("No models could be tested successfully.")

if __name__ == "__main__":
    main()