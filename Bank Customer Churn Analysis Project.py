# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv(r"E:\Churn_Modelling (1).csv")

# Check the column names to ensure correctness
print(df.columns)

# Part 1: Data Loading and Basic Python Operations

# Function to calculate average balance for different age groups
def calculate_age_group_balance(df):
    # Create age groups using conditional statements
    df['age_group'] = None
    for idx in df.index:
        age = df.loc[idx, 'Age']
        if age < 30:
            df.loc[idx, 'age_group'] = 'Young'
        elif age < 50:
            df.loc[idx, 'age_group'] = 'Middle-aged'
        else:
            df.loc[idx, 'age_group'] = 'Senior'
    # Calculate average balance per age group
    return df.groupby('age_group')['Balance'].mean()

# Calculate average balance by age group
age_group_balance = calculate_age_group_balance(df)
print(age_group_balance)

# Part 2: Data Structure Manipulation

# Create lists of churned and retained customers
churned_customers = df[df['Exited'] == 1]['CustomerId'].tolist()
retained_customers = df[df['Exited'] == 0]['CustomerId'].tolist()

# Example list comprehension for high-value customers (Balance > 100,000)
high_value_customers = [
    customer_id for customer_id, balance
    in zip(df['CustomerId'], df['Balance'])
    if balance > 100000
]

# Create a dictionary with customer statistics by country
country_stats = {
    country: {
        'avg_balance': df[df['Geography'] == country]['Balance'].mean(),
        'churn_rate': df[df['Geography'] == country]['Exited'].mean() * 100
    }
    for country in df['Geography'].unique()
}

print(country_stats)

# Part 3: Data Cleaning and Preparation

# Function to prepare data
def prepare_data(df):
    # Handle missing values
    df['Balance'].fillna(df['Balance'].mean(), inplace=True)
    # Create new features
    df['balance_per_product'] = df['Balance'] / df['NumOfProducts']
    df['is_high_value'] = df['Balance'] > df['Balance'].mean()
    # Convert categorical variables
    df = pd.get_dummies(df, columns=['Gender', 'Geography'])
    return df

# Prepare the data
df = prepare_data(df)

# Part 4: Exploratory Data Analysis and Visualization

# Function to create visualizations
def create_visualizations(df):
    # Set up the matplotlib figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Age distribution by churn status
    sns.histplot(data=df, x='Age', hue='Exited', ax=axes[0, 0])
    axes[0, 0].set_title('Age Distribution by Churn Status')

    # Balance distribution by product number
    sns.boxplot(data=df, x='NumOfProducts', y='Balance', ax=axes[0, 1])
    axes[0, 1].set_title('Balance Distribution by Product Number')

    # Correlation heatmap
    numeric_cols = ['Age', 'Balance', 'CreditScore', 'Tenure']
    sns.heatmap(df[numeric_cols].corr(), annot=True, ax=axes[1, 0])
    axes[1, 0].set_title('Correlation Heatmap')

    # Churn rate by credit score range
    df['CreditScoreRange'] = pd.cut(df['CreditScore'], bins=[300, 500, 600, 700, 800, 850], labels=['300-500', '500-600', '600-700', '700-800', '800-850'])
    sns.barplot(data=df, x='CreditScoreRange', y='Exited', ax=axes[1, 1])
    axes[1, 1].set_title('Churn Rate by Credit Score Range')

    plt.tight_layout()
    return fig

# Create and display the visualizations
fig = create_visualizations(df)
plt.show()

# Part 5: Basic Predictive Analysis

# Prepare features for modeling
df_model = df[['CreditScore', 'Age', 'Balance', 'NumOfProducts', 'IsActiveMember']]

# Split data into training and testing sets
from sklearn.model_selection import train_test_split

X = df_model
y = df['Exited']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a simple Logistic Regression model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

model = LogisticRegression()
model.fit(X_train, y_train)

# Predict the test set
y_pred = model.predict(X_test)

# Print the classification report
print(classification_report(y_test, y_pred))

# Print confusion matrix
print(confusion_matrix(y_test, y_pred))

