import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Load the data
train_df = pd.read_csv(r"D:\Kabil\Skillvertex\train.csv")
test_df = pd.read_csv(r"D:\Kabil\Skillvertex\test.csv")

# Display info about the dataframes
print(train_df.info())
print(test_df.info())

# Check for missing values
print(train_df.isna().sum())
print(test_df.isna().sum())

# Replace "?" with NaN
train_df.replace("?", np.nan, inplace=True)
test_df.replace("?", np.nan, inplace=True)

# Drop the 'Loan_ID' column
train_df = train_df.drop(columns=["Loan_ID"])
test_df = test_df.drop(columns=["Loan_ID"])

# Encode categorical variables
label_encoder = LabelEncoder()
for column in train_df.columns:
    if train_df[column].dtype == 'object':
        train_df[column] = label_encoder.fit_transform(train_df[column])

for column in test_df.columns:
    if test_df[column].dtype == 'object':
        test_df[column] = label_encoder.fit_transform(test_df[column])

# Select features and target
selected_cols = ["ApplicantIncome", "CoapplicantIncome", "Loan_Amount_Term", "Property_Area"]
X_train = train_df[selected_cols]
y_train = train_df["LoanAmount"]

# Ensure Property_Area is numerical for imputation
X_train['Property_Area'] = label_encoder.fit_transform(X_train['Property_Area'])
X_test = test_df[selected_cols]
X_test['Property_Area'] = label_encoder.transform(X_test['Property_Area'])

# Impute missing values
imputer_X = SimpleImputer(strategy="median")
imputer_y = SimpleImputer(strategy="median")
X_train_imputed = imputer_X.fit_transform(X_train)
y_train_imputed = imputer_y.fit_transform(y_train.values.reshape(-1, 1)).flatten()

# Process test data
y_test = test_df["LoanAmount"]
X_test_imputed = imputer_X.transform(X_test)
y_test_imputed = imputer_y.transform(y_test.values.reshape(-1, 1)).flatten()

# Split the data into training and testing sets
X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X_train_imputed, y_train_imputed, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train_final, y_train_final)

# Make predictions
y_pred = model.predict(X_test_final)

# Evaluate the model
mse = mean_squared_error(y_test_final, y_pred)
print("Mean Squared Error:", mse)

r2 = r2_score(y_test_final, y_pred)
print("R-squared:", r2 )
print("R-squared percentage:" , r2*100) 

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(2, 2, 1)
plt.scatter(X_train_final[:, 2], y_train_final, color='orange', label='Data Points')
plt.xlabel('Loan_Amount_Term')
plt.ylabel('LoanAmount')
plt.title('Loan_Amount_Term vs. LoanAmount')
plt.grid(True)
plt.tight_layout()
plt.show()
