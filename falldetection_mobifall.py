#!/usr/bin/env python3
"""
Fall Detection Models - Modified for MobiFall Dataset
This script adapts the original code to work with the MobiFall dataset structure
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import time
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler
import logging

# Setup basic configuration for logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directories for saving models and plots
plot_dir = 'plots'
model_dir = 'models'

# Create directories if they do not exist
os.makedirs(plot_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)

# Dataset path - updated for MobiFall dataset
dataset_path = 'fall/MobiFall_Dataset_v2.0'

# Define fall and ADL types (matching MobiFall dataset)
fall_types = ['FOL', 'FKL', 'BSC', 'SDL']
adl_types = ['STD', 'WAL', 'JOG', 'JUM', 'STU', 'STN', 'SCH', 'CSI', 'CSO']

def load_mobifall_data(file_path):
    """Load data from MobiFall text files"""
    try:
        # Read the file and skip header lines
        with open(file_path, 'r') as f:
            lines = f.readlines()
        
        # Find the @DATA line
        data_start = None
        for i, line in enumerate(lines):
            if line.strip() == '@DATA':
                data_start = i + 1
                break
        
        if data_start is None:
            return None
        
        # Parse data lines
        data = []
        for line in lines[data_start:]:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split(',')
                if len(parts) >= 4:
                    # timestamp, x, y, z
                    timestamp = float(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    z = float(parts[3])
                    data.append([timestamp, x, y, z])
        
        return np.array(data)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def combine_acc_gyro_data(acc_data, gyro_data):
    """Combine accelerometer and gyroscope data"""
    if acc_data is None or gyro_data is None:
        return None
    
    # Use accelerometer timestamps as reference
    acc_timestamps = acc_data[:, 0]
    gyro_timestamps = gyro_data[:, 0]
    
    # Find common time range
    start_time = max(acc_timestamps[0], gyro_timestamps[0])
    end_time = min(acc_timestamps[-1], gyro_timestamps[-1])
    
    # Filter data to common time range
    acc_mask = (acc_timestamps >= start_time) & (acc_timestamps <= end_time)
    gyro_mask = (gyro_timestamps >= start_time) & (gyro_timestamps <= end_time)
    
    acc_filtered = acc_data[acc_mask]
    gyro_filtered = gyro_data[gyro_mask]
    
    # Take minimum length to ensure same size
    min_len = min(len(acc_filtered), len(gyro_filtered))
    
    if min_len < 100:  # Need at least 100 samples
        return None
    
    # Combine data: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    combined = np.column_stack([
        acc_filtered[:min_len, 1:4],  # acc x,y,z
        gyro_filtered[:min_len, 1:4]  # gyro x,y,z
    ])
    
    return combined

def load_mobifall_dataset():
    """Load the complete MobiFall dataset"""
    print("Loading MobiFall dataset...")
    
    xtrain = []
    ytrain = []
    
    # Get list of subjects
    subjects = [d for d in os.listdir(dataset_path) if d.startswith('sub')]
    subjects.sort()
    
    print(f"Found {len(subjects)} subjects")
    
    # Load ADL data (label 0)
    print("Loading ADL data...")
    adl_count = 0
    for subject in subjects:
        subject_path = os.path.join(dataset_path, subject)
        adl_path = os.path.join(subject_path, 'ADL')
        
        if not os.path.exists(adl_path):
            continue
            
        for adl_type in adl_types:
            adl_type_path = os.path.join(adl_path, adl_type)
            if not os.path.exists(adl_type_path):
                continue
                
            # Get all files for this ADL type
            files = os.listdir(adl_type_path)
            acc_files = [f for f in files if f.endswith('_acc_') or '_acc_' in f]
            
            for acc_file in acc_files:
                # Find corresponding gyro file
                gyro_file = acc_file.replace('_acc_', '_gyro_')
                
                acc_path = os.path.join(adl_type_path, acc_file)
                gyro_path = os.path.join(adl_type_path, gyro_file)
                
                if os.path.exists(gyro_path):
                    acc_data = load_mobifall_data(acc_path)
                    gyro_data = load_mobifall_data(gyro_path)
                    
                    combined_data = combine_acc_gyro_data(acc_data, gyro_data)
                    
                    if combined_data is not None and len(combined_data) >= 600:
                        # Create windows of 600 samples
                        for start_idx in range(0, len(combined_data) - 600 + 1, 300):
                            window = combined_data[start_idx:start_idx + 600]
                            xtrain.append(window)
                            ytrain.append(0)  # ADL label
                            adl_count += 1
                            
                            # Limit ADL samples to balance dataset
                            if adl_count >= 1000:
                                break
                if adl_count >= 1000:
                    break
            if adl_count >= 1000:
                break
        if adl_count >= 1000:
            break
    
    print(f"Loaded {adl_count} ADL samples")
    
    # Load Fall data (label 1)
    print("Loading Fall data...")
    fall_count = 0
    for subject in subjects:
        subject_path = os.path.join(dataset_path, subject)
        falls_path = os.path.join(subject_path, 'FALLS')
        
        if not os.path.exists(falls_path):
            continue
            
        for fall_type in fall_types:
            fall_type_path = os.path.join(falls_path, fall_type)
            if not os.path.exists(fall_type_path):
                continue
                
            # Get all files for this fall type
            files = os.listdir(fall_type_path)
            acc_files = [f for f in files if '_acc_' in f]
            
            for acc_file in acc_files:
                # Find corresponding gyro file
                gyro_file = acc_file.replace('_acc_', '_gyro_')
                
                acc_path = os.path.join(fall_type_path, acc_file)
                gyro_path = os.path.join(fall_type_path, gyro_file)
                
                if os.path.exists(gyro_path):
                    acc_data = load_mobifall_data(acc_path)
                    gyro_data = load_mobifall_data(gyro_path)
                    
                    combined_data = combine_acc_gyro_data(acc_data, gyro_data)
                    
                    if combined_data is not None and len(combined_data) >= 600:
                        # For falls, create multiple windows around the event
                        # Assume fall happens in the middle of the recording
                        mid_point = len(combined_data) // 2
                        
                        # Create windows around the fall event
                        for offset in [-300, -150, 0, 150]:
                            start_idx = max(0, mid_point + offset - 300)
                            end_idx = start_idx + 600
                            
                            if end_idx <= len(combined_data):
                                window = combined_data[start_idx:end_idx]
                                xtrain.append(window)
                                ytrain.append(1)  # Fall label
                                fall_count += 1
    
    print(f"Loaded {fall_count} Fall samples")
    
    return np.array(xtrain), np.array(ytrain)

def normalize_data(X_train, X_val, X_test):
    """Normalize data using min-max normalization"""
    print("Normalizing data...")
    
    for i in range(X_train.shape[2]):
        min_val = np.min(X_train[:, :, i])
        max_val = np.max(X_train[:, :, i])
        
        if max_val > min_val:  # Avoid division by zero
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
    
    # Add early stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )
    
    # Train model
    if 'bilstm_cnn' in model_name.lower():
        # For multi-input model
        history = model.fit(
            [X_train[:, :, :3], X_train[:, :, 3:6]], y_train,
            validation_data=([X_val[:, :, :3], X_val[:, :, 3:6]], y_val),
            epochs=epochs, batch_size=batch_size, verbose=1,
            callbacks=[early_stopping]
        )
        
        # Evaluate
        test_loss, test_acc = model.evaluate([X_test[:, :, :3], X_test[:, :, 3:6]], y_test, verbose=0)
        y_pred = model.predict([X_test[:, :, :3], X_test[:, :, 3:6]])
        
    else:
        # For single-input model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs, batch_size=batch_size, verbose=1,
            callbacks=[early_stopping]
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
    plt.savefig(os.path.join(plot_dir, f'{model_name}_mobifall_training_history.png'))
    plt.show()
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f'{model_name} - Confusion Matrix (MobiFall)')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(os.path.join(plot_dir, f'{model_name}_mobifall_confusion_matrix.png'))
    plt.show()
    
    # Classification Report
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred_binary))
    
    # Save model
    model_path = os.path.join(model_dir, f'{model_name}_mobifall_model.h5')
    model.save(model_path)
    print(f"Model saved to: {model_path}")
    
    return history, test_acc, test_loss

def main():
    """Main function to run the MobiFall dataset training"""
    print("Fall Detection Models - MobiFall Dataset")
    print("=" * 50)
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        print("Please ensure the MobiFall dataset is in the 'fall' directory")
        return
    
    # Load data
    try:
        X, y = load_mobifall_dataset()
        print(f"Loaded data shape: X={X.shape}, y={y.shape}")
        
        if len(X) == 0:
            print("No data loaded. Please check the dataset structure.")
            return
            
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Class distribution - Train: {np.bincount(y_train)}, Val: {np.bincount(y_val)}, Test: {np.bincount(y_test)}")
    
    # Normalize data
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)
    
    # Test models
    results = {}
    
    # 1. Simple LSTM
    print("\n" + "="*50)
    lstm_model = build_simple_lstm_model((X_train.shape[1], X_train.shape[2]))
    history, acc, loss = train_and_evaluate_model(
        lstm_model, X_train, y_train, X_val, y_val, X_test, y_test, 
        "Simple_LSTM", epochs=30
    )
    results['Simple_LSTM'] = {'accuracy': acc, 'loss': loss}
    
    # 2. Bi-LSTM + CNN
    print("\n" + "="*50)
    bilstm_cnn_model = build_bilstm_cnn_model((X_train.shape[1], X_train.shape[2]))
    history, acc, loss = train_and_evaluate_model(
        bilstm_cnn_model, X_train, y_train, X_val, y_val, X_test, y_test, 
        "BiLSTM_CNN", epochs=30
    )
    results['BiLSTM_CNN'] = {'accuracy': acc, 'loss': loss}
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS SUMMARY - MobiFall Dataset")
    print("="*60)
    
    for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        print(f"{model_name:15} - Accuracy: {metrics['accuracy']:.4f}, Loss: {metrics['loss']:.4f}")
    
    print(f"\nAll models and plots saved to:")
    print(f"Models: {model_dir}/")
    print(f"Plots: {plot_dir}/")

if __name__ == "__main__":
    main()