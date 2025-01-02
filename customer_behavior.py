import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("E-commerce Customer Behavior - Sheet1.csv")

# Descriptive statistics
print(df.head())
print(df.describe())
print(df.info())
print(df.columns)

# Elimination of null values
df = df.dropna(axis= 0)

cat_columns = ['Gender', 'City', 'Membership Type','Discount Applied', 'Satisfaction Level']

num_columns = ['Age','Total Spend', 'Items Purchased', 'Average Rating','Days Since Last Purchase']

#  Visualization of categorical and numerical variables
for column in cat_columns:
    print(df[column].value_counts())

for column in cat_columns:
    ax = df[column].value_counts().plot(kind="bar", color=sns.color_palette("pastel"))

    for i, v in enumerate(df[column].value_counts()):
        ax.text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=10)

    plt.title(f"Values by {column}")
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(f"{column}")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

for column in num_columns:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=df[column], color='lightblue', width=0.5)

    plt.title(f"Box Plot of {column}", fontsize=14)
    plt.xlabel(f"{column}", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# Total Spend by Categorical Variable
for column in cat_columns:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x=column, y='Total Spend', data=df, palette='Set2')

    plt.title(f'Total Spend by {column}', fontsize=14)
    plt.xlabel(f'{column}', fontsize=12)
    plt.ylabel('Total Spend', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()

# Correlation Matrix
corr_matrix = df.corr()
print(corr_matrix)

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=1)
plt.title("Correlation Matrix")
plt.show()

# Total Spend by Days since last purchase
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Days Since Last Purchase', y='Total Spend', data=df,
                s=100,
                hue='Satisfaction Level',
                palette='viridis',
                edgecolor='black',
                alpha=0.7)

sns.regplot(x='Days Since Last Purchase', y='Total Spend', data=df, scatter=False, color='red')

plt.title('Total Spend vs Days Since Last Purchase', fontsize=14)
plt.xlabel('Days Since Last Purchase', fontsize=12)
plt.ylabel('Total Spend', fontsize=12)

plt.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()