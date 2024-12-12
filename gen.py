import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load the original data
original_data = pd.read_csv('data.csv')

# Define the features we want to generate
features = [
    'Recharge from rainfallMonsoon season', 'Recharge from other sources',
    'Recharge from rainfallNon-monsoon season', 'Total_Rainfall',
    'Natural discharge during non-monsoon season', 'Net annual groundwater availability',
    'Irrigation', 'Domestic and industrial uses', 'Total_Usage',
    'Projected demand for domestic and industrial uses upto 2025',
    'Groundwater availability for future irrigation use'
]

# Function to generate synthetic data
def generate_synthetic_data(original_data, num_samples):
    synthetic_data = pd.DataFrame(columns=original_data.columns)
    
    # Generate synthetic data for each feature
    for feature in features:
        min_val = original_data[feature].min()
        max_val = original_data[feature].max()
        mean = original_data[feature].mean()
        std = original_data[feature].std()
        
        # Generate random values following a normal distribution
        synthetic_values = np.random.normal(mean, std, num_samples)
        
        # Clip values to be within the original range
        synthetic_values = np.clip(synthetic_values, min_val, max_val)
        
        synthetic_data[feature] = synthetic_values
    
    # Calculate Total_Rainfall
    synthetic_data['Total_Rainfall'] = (
        synthetic_data['Recharge from rainfallMonsoon season'] +
        synthetic_data['Recharge from other sources'] +
        synthetic_data['Recharge from rainfallNon-monsoon season']
    )
    
    # Calculate Net annual groundwater availability
    synthetic_data['Net annual groundwater availability'] = (
        synthetic_data['Total_Rainfall'] -
        synthetic_data['Natural discharge during non-monsoon season']
    )
    
    # Calculate Total_Usage
    synthetic_data['Total_Usage'] = (
        synthetic_data['Irrigation'] +
        synthetic_data['Domestic and industrial uses']
    )
    
    # Calculate Groundwater availability for future irrigation use
    synthetic_data['Groundwater availability for future irrigation use'] = (
        synthetic_data['Net annual groundwater availability'] -
        synthetic_data['Total_Usage'] -
        synthetic_data['Projected demand for domestic and industrial uses upto 2025']
    )
    
    # Determine the Situation based on the generated data
    conditions = [
        (synthetic_data['Groundwater availability for future irrigation use'] > 10),
        (synthetic_data['Groundwater availability for future irrigation use'] > 5) & (synthetic_data['Groundwater availability for future irrigation use'] <= 10),
        (synthetic_data['Groundwater availability for future irrigation use'] > 0) & (synthetic_data['Groundwater availability for future irrigation use'] <= 5),
        (synthetic_data['Groundwater availability for future irrigation use'] <= 0)
    ]
    choices = ['EXCESS', 'MODERATED', 'SEMICRITICAL', 'CRITICAL']
    synthetic_data['Situation'] = np.select(conditions, choices, default='UNKNOWN')
    
    # Generate synthetic state names
    synthetic_data['States'] = [f'Synthetic State {i+1}' for i in range(num_samples)]
    
    return synthetic_data

# Generate synthetic data (e.g., 100 new samples)
num_new_samples = 100
synthetic_data = generate_synthetic_data(original_data, num_new_samples)

# Combine original and synthetic data
combined_data = pd.concat([original_data, synthetic_data], ignore_index=True)

# Save the combined data to a new CSV file
combined_data.to_csv('combined_data.csv', index=False)

print(f"Generated {num_new_samples} new synthetic samples.")
print(f"Combined data saved to 'combined_data.csv' with {len(combined_data)} total samples.")