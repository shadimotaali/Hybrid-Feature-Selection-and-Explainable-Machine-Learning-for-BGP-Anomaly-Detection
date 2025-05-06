import pandas as pd

# Load your dataset
file_path = 'bgp-feature-extractor/Randomly_Balanced_Dataset.xlsx'

# Read the Excel file
df = pd.read_excel(file_path)

# Select only numeric columns
numeric_df = df.select_dtypes(include=['number'])

# Calculate variances for numeric columns
variances = numeric_df.var()

# Display the results
print("Variance for each numeric column:")
print(variances)

# Save the variances to a new Excel file
output_path = 'bgp-feature-extractor/column_variances.xlsx'
variances.to_excel(output_path)

print(f"\nVariances saved to: {output_path}")
