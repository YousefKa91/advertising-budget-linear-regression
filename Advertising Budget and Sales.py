# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Set working directory to script location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load dataset
try:
    df = pd.read_csv('Advertising Budget and Sales.csv')
except FileNotFoundError:
    print("Error: 'Advertising Budget and Sales.csv' not found.")
    exit()

# Clean up index column if present
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

sns.set_style("whitegrid")
print("Data loaded successfully. Starting EDA...")

# --- Exploratory Data Analysis ---

# Distribution of Sales
plt.figure(figsize=(12, 6))
sns.histplot(df['Sales ($)'], kde=True, bins=30, color='skyblue')
plt.title('Distribution of Sales', fontsize=16)
plt.xlabel('Sales ($)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Advertising Budget and Sales', fontsize=16)
plt.show()

# Pairplot to see relationships
sns.pairplot(df, diag_kind='kde', palette='viridis')
plt.suptitle('Pairplot of Advertising Budgets and Sales', y=1.02, fontsize=16)
plt.show()

# Individual scatter plots for each channel
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
channels = ['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)']
colors = ['coral', 'lightblue', 'lightgreen']

for idx, (channel, color) in enumerate(zip(channels, colors)):
    axes[idx].scatter(df[channel], df['Sales ($)'], alpha=0.6, color=color)
    axes[idx].set_xlabel(channel, fontsize=12)
    axes[idx].set_ylabel('Sales ($)', fontsize=12)
    axes[idx].set_title(f'{channel} vs Sales', fontsize=14)
    axes[idx].grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- Model Preparation ---

print("\nEDA complete. Preparing model...")

# Define features and target
X = df[['TV Ad Budget ($)', 'Radio Ad Budget ($)', 'Newspaper Ad Budget ($)']]
y = df['Sales ($)']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Model Training ---

# Pipeline with scaling and linear regression
model_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('regressor', LinearRegression())
])

print("\nTraining model...")
model_pipeline.fit(X_train, y_train)
print("Training complete!")

# Extract coefficients
regressor = model_pipeline.named_steps['regressor']
coefficients = regressor.coef_
intercept = regressor.intercept_

print("\n--- Model Coefficients ---")
print(f"Intercept: {intercept:.3f}")
for feature, coef in zip(X.columns, coefficients):
    print(f"{feature}: {coef:.4f}")

# --- Model Evaluation ---

print("\nEvaluating on test set...")
y_pred = model_pipeline.predict(X_test)

# Performance metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
print(f"\nModel Performance:")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# Actual vs Predicted
plt.figure(figsize=(10, 10))
plt.scatter(y_test, y_pred, alpha=0.6, color='purple', s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r', linewidth=2)
plt.title('Actual vs. Predicted Sales', fontsize=16)
plt.xlabel('Actual Sales ($)', fontsize=12)
plt.ylabel('Predicted Sales ($)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Residuals plot
residuals = y_test - y_pred
plt.figure(figsize=(12, 6))
plt.scatter(y_pred, residuals, alpha=0.6, color='orange')
plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
plt.title('Residuals vs. Predicted Values', fontsize=16)
plt.xlabel('Predicted Sales ($)', fontsize=12)
plt.ylabel('Residuals', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Residuals distribution
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True, bins=20, color='teal')
plt.title('Distribution of Residuals', fontsize=16)
plt.xlabel('Residuals', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.show()

# Feature importance (coefficients)
plt.figure(figsize=(10, 6))
features = X.columns
plt.barh(features, coefficients, color=['coral', 'lightblue', 'lightgreen'], alpha=0.8, edgecolor='black')
plt.xlabel('Coefficient Value', fontsize=12)
plt.ylabel('Advertising Channel', fontsize=12)
plt.title('Linear Regression Coefficients', fontsize=16)
plt.axvline(x=0, color='black', linestyle='--', linewidth=1)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

print("\nAnalysis complete!")
