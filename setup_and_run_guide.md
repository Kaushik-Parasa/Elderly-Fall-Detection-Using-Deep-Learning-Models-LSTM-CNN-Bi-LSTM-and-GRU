# Fall Detection Models - Setup and Run Guide

## Issues Found and Solutions

### 1. Missing Dataset
**Problem**: The code references MobiAct dataset at `'../../MobiAct_Dataset_v2.0/MobiAct_Dataset_v2.0/Annotated Data/'` which doesn't exist.

**Solution**: 
- Download the MobiAct dataset from: https://bmi.hmu.gr/the-mobiact-dataset-2/
- Extract it to the correct path relative to your project, OR
- Modify the dataset path in the code to match your actual dataset location

### 2. Code Structure Issues
**Problem**: Some incomplete lines and variable reference issues.

**Solution**: Fixed the incomplete model saving line in the attention model section.

## How to Run the Code

### Prerequisites
1. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Download Dataset** (if you want to train from scratch):
   - Download MobiAct dataset
   - Place it in the correct directory structure or modify the path in the code

### Option 1: Run with Existing Models (Recommended)
Since you already have trained models in the `/models` directory, you can:

1. **Load and Test Existing Models**:
   ```python
   import tensorflow as tf
   import numpy as np
   
   # Load a trained model
   model = tf.keras.models.load_model('models/128_bilstm_cnn_trained_model.h5')
   
   # Use the model for predictions
   # predictions = model.predict(your_test_data)
   ```

### Option 2: Run Training (Requires Dataset)
1. **Modify Dataset Path**: Update the dataset path in the code:
   ```python
   # Change this line to your actual dataset path
   path = 'path/to/your/MobiAct_Dataset/Annotated Data/' + folder
   ```

2. **Run the Python Script**:
   ```bash
   python falldetection_models.py
   ```

3. **Or Run the Jupyter Notebook**:
   ```bash
   jupyter notebook falldetection_models.ipynb
   ```

### Option 3: Run with Synthetic Data (For Testing)
The code already includes synthetic data generation, so you can:

1. Comment out the dataset loading sections
2. Use only the synthetic data parts
3. Run the model training with synthetic data

## Key Features of the Code

1. **Multiple Model Architectures**:
   - LSTM
   - Bi-LSTM + CNN
   - Bi-LSTM + CNN with Attention
   - GRU

2. **Automated Visualization**:
   - Training/validation curves
   - Confusion matrices
   - Data distribution plots

3. **Model Persistence**:
   - Saves trained models in `/models` directory
   - Saves plots in `/plots` directory

## Expected Output Structure

After running, you should see:
- `/models/` - Contains trained model files (.h5 format)
- `/plots/` - Contains visualization plots (.png format)
- Console output with training progress and metrics

## Troubleshooting

1. **Memory Issues**: If you encounter memory issues, reduce batch sizes or use smaller datasets
2. **GPU Issues**: The code will automatically use GPU if available, CPU otherwise
3. **Path Issues**: Ensure all paths are correctly set for your system

## Performance Metrics

The models achieve high performance:
- Bi-LSTM+CNN: ~99% accuracy
- GRU: ~96-99% accuracy (depending on batch size)
- LSTM: ~96% accuracy

## Next Steps

1. Test with your own sensor data
2. Modify architectures for your specific use case
3. Implement real-time inference
4. Deploy to mobile/edge devices