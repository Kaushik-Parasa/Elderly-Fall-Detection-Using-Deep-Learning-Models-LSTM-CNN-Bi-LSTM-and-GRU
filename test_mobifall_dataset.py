#!/usr/bin/env python3
"""
Test script to verify MobiFall dataset structure and loading
"""

import os
import numpy as np

def test_mobifall_structure():
    """Test the MobiFall dataset structure"""
    dataset_path = 'fall/MobiFall_Dataset_v2.0'
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return False
    
    print("Testing MobiFall dataset structure...")
    
    # Get list of subjects
    subjects = [d for d in os.listdir(dataset_path) if d.startswith('sub')]
    subjects.sort()
    
    print(f"Found {len(subjects)} subjects: {subjects[:5]}...")
    
    # Check a few subjects for structure
    adl_count = 0
    fall_count = 0
    
    for i, subject in enumerate(subjects[:5]):  # Test first 5 subjects
        subject_path = os.path.join(dataset_path, subject)
        print(f"\nChecking {subject}:")
        
        # Check ADL folder
        adl_path = os.path.join(subject_path, 'ADL')
        if os.path.exists(adl_path):
            adl_activities = os.listdir(adl_path)
            print(f"  ADL activities: {adl_activities}")
            
            # Check one ADL activity
            if adl_activities:
                activity_path = os.path.join(adl_path, adl_activities[0])
                files = os.listdir(activity_path)
                acc_files = [f for f in files if '_acc_' in f]
                gyro_files = [f for f in files if '_gyro_' in f]
                print(f"    {adl_activities[0]}: {len(acc_files)} acc files, {len(gyro_files)} gyro files")
                adl_count += len(acc_files)
        
        # Check FALLS folder
        falls_path = os.path.join(subject_path, 'FALLS')
        if os.path.exists(falls_path):
            fall_activities = os.listdir(falls_path)
            print(f"  Fall activities: {fall_activities}")
            
            # Check one fall activity
            if fall_activities:
                activity_path = os.path.join(falls_path, fall_activities[0])
                files = os.listdir(activity_path)
                acc_files = [f for f in files if '_acc_' in f]
                gyro_files = [f for f in files if '_gyro_' in f]
                print(f"    {fall_activities[0]}: {len(acc_files)} acc files, {len(gyro_files)} gyro files")
                fall_count += len(acc_files)
    
    print(f"\nTotal files found (first 5 subjects): ADL={adl_count}, Falls={fall_count}")
    return True

def load_sample_file():
    """Load and examine a sample data file"""
    print("\nTesting file loading...")
    
    # Try to find a sample file
    dataset_path = 'fall/MobiFall_Dataset_v2.0'
    subjects = [d for d in os.listdir(dataset_path) if d.startswith('sub')]
    
    for subject in subjects[:3]:
        subject_path = os.path.join(dataset_path, subject)
        falls_path = os.path.join(subject_path, 'FALLS')
        
        if os.path.exists(falls_path):
            fall_types = os.listdir(falls_path)
            if fall_types:
                fall_type_path = os.path.join(falls_path, fall_types[0])
                files = os.listdir(fall_type_path)
                acc_files = [f for f in files if '_acc_' in f]
                
                if acc_files:
                    sample_file = os.path.join(fall_type_path, acc_files[0])
                    print(f"Loading sample file: {sample_file}")
                    
                    try:
                        with open(sample_file, 'r') as f:
                            lines = f.readlines()
                        
                        print(f"File has {len(lines)} lines")
                        
                        # Find data section
                        data_start = None
                        for i, line in enumerate(lines):
                            if line.strip() == '@DATA':
                                data_start = i + 1
                                break
                        
                        if data_start:
                            print(f"Data starts at line {data_start}")
                            
                            # Parse a few data lines
                            data_lines = []
                            for line in lines[data_start:data_start+10]:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    parts = line.split(',')
                                    if len(parts) >= 4:
                                        data_lines.append([float(p.strip()) for p in parts[:4]])
                            
                            if data_lines:
                                data_array = np.array(data_lines)
                                print(f"Sample data shape: {data_array.shape}")
                                print(f"Sample data:\n{data_array}")
                                return True
                        
                    except Exception as e:
                        print(f"Error loading file: {e}")
    
    return False

if __name__ == "__main__":
    print("MobiFall Dataset Test")
    print("=" * 30)
    
    if test_mobifall_structure():
        load_sample_file()
    else:
        print("Dataset structure test failed")