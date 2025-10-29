#!/usr/bin/env python3
"""
Test script for Fall Detection Models
This script can run without the full MobiAct dataset by using synthetic data
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Setup directories
plot_dir = 'plots'
model_dir = 'models'

# Create directories if they don't exist
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

def generate_synthetic_data(n_samples=1000, sequence_length=600, n_features=6):
    """Generate synthetic sensor data for testing"""
    print("Generating synthetic data...")
    
    # Generate synthetic fall data (class 1)
    fall_data = []
    for i in range(n_samples // 2):
        # Simulate fall pattern with high acceleration spike
        data = np.random.randn(sequence_length, n_features) * 0.5
        # Add fall signature (sudden spike in acceleration)
        spike_start = np.random.randint(100, 400)
        data[spike_start:spike_start+50, :3] += np.random.randn(50, 3) * 3  # High acceleration
        fall_data.append(data)
    
    # Generate synthetic ADL data (class 0)
    adl_data = []
    for i in range(n_samples // 2):
        # Normal activity patterns
        data = np.random.randn(sequence_length, n_features) * 0.8
        adl_data.append(data)
    
    # Combine data
    X = np.array(fall_data + adl_data)
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples // 2))
    
    return X, y

def normalize_data(X_train, X_val, X_test):
    """Normalize data using min-max normalization"""
    print("Normalizing data...")
    
    for i in range(X_train.shape[2]):
        min_val = np.min(X_train[:, :, i])
        max_val = np.max(X_train[:, :, i])
        
        X_train[:, :, i] = 2 * (X_train[:, :, i] - min_val) / (max_val - min_val) - 1
        X_val[:, :, i] = 2 * (X_val[:, :, i] - min_val) / (max_val - min_val) - 1
        X_test[:, :, i] = 2 * (X_test[:, :, i] - min_val) / (max_val - min_val) - 1
    
    return X_train, X_val, X_test

def build_simple_lstm_model(input_shape):
    """Build a simple LSTM model"""
    model = keras.Sequential([
        layers.LSTM(64, input_shape=input_shape),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

def build_bilstm_cnn_model(input_shape):
    """Build Bi-LSTM + CNN model"""
    input_layer1 = keras.Input((input_shape[0], 3), name='Input1')
    input_layer2 = keras.Input((input_shape[0], 3), name='Input2')
    
    # CNN branches
    def cnn_branch(input_layer):
        cnn = layers.Conv1D(16, 7, padding="same")(input_layer)
        cnn = layers.BatchNormalization()(cnn)
        cnn = layers.Dropout(0.2)(cnn)
        
        cnn = layers.Conv1D(32, 5, padding="same")(cnn)
        cnn = layers.BatchNormalization()(cnn)
        cnn = layers.Dropout(0.2)(cnn)
        
        return cnn
    
    output1 = cnn_branch(input_layer1)
    output2 = cnn_branch(input_layer2)
    
    # Concatenate CNN outputs
    output = layers.concatenate([output1, output2])
    
    # Bi-LSTM layers
    lstm = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(output)
    lstm = layers.LayerNormalization()(lstm)
    lstm = layers.Dropout(0.2)(lstm)
    
    lstm = layers.Bidirectional(layers.LSTM(64))(lstm)
    lstm = layers.LayerNormalization()(lstm)
    lstm = layers.Dropout(0.2)(lstm)
    
    # Dense layers
    dense = layers.Dense(32, activation='relu')(lstm)
    output = layers.Dense(1, activation='sigmoid')(dense)
    
    return keras.Model([input_layer1, input_layer2], output)

def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, X_test, y_test, 
                           model_name, epochs=20, batch_size=32):
    """Train and evaluate a model"""
    print(f"\nTraining {model_name}...")
    
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    # Train model
    if 'bilstm_cnn' in model_name.lower():
        # For multi-input model
        history = model.fit(
            [X_train[:, :, :3], X_train[:, :, 3:6]], y_train,
            validation_data=([X_val[:, :, :3], X_val[:, :, 3:6]], y_val),
            epochs=epochs, batch_size=batch_size, verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate([X_test[:, :, :3], X_test[:, :, 3:6]], y_test, verbose=0)
        y_pred = model.predict([X_test[:, :, :3], X_test[:, :, 3:6]])
        
    else:
        # For single-input model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, verbose=1
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        y_pred = model.predict(X_test)
    
    # Convert predictions to binary
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    print(f"{model_name} - Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{model_name}_training_history.png'))
    plt.show()
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'{model_name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(plot_dir, f'{model_name}_confusion_matrix.png'))
    plt.show()
    
    # Classification Report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred_binary))
    
    # Save model
    model_path = os.path.join(model_dir, f'{model_name}_model.h5')
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    return history, test_acc, test_loss

def main():
    """Main function to run the test"""
    print("Fall Detection Models - Test Script")
    print("=" * 50)
    
    # Generate synthetic data
    X, y = generate_synthetic_data(n_samples=2000)
    print(f"Generated data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Normalize data
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)
    
    # Test models
    results = {}
    
    # 1. Simple LSTM
    print("\n" + "="*50)
    lstm_model = build_simple_lstm_model((X_train.shape[1], X_train.shape[2]))
    history, acc, loss = train_and_evaluate_model(
        lstm_model, X_train, y_train, X_val, y_val, X_test, y_test, 
        "Simple_LSTM", epochs=15
    )
    results['Simple_LSTM'] = {'accuracy': acc, 'loss': loss}
    
    # 2. Bi-LSTM + CNN
    print("\n" + "="*50)
    bilstm_cnn_model = build_bilstm_cnn_model((X_train.shape[1], X_train.shape[2]))
    history, acc, loss = train_and_evaluate_model(
        bilstm_cnn_model, X_train, y_train, X_val, y_val, X_test, y_test, 
        "BiLSTM_CNN", epochs=15
    )
    results['BiLSTM_CNN'] = {'accuracy': acc, 'loss': loss}
    
    # Summary
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    for model_name, metrics in results.items():
        print(f"{model_name:15} - Accuracy: {metrics['accuracy']:.4f}, Loss: {metrics['loss']:.4f}")
    
    print(f"\nAll models and plots saved to:")
    print(f"Models: {model_dir}/")
    print(f"Plots: {plot_dir}/")

if __name__ == "__main__":
    main()