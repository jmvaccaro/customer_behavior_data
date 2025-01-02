import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency

df = pd.read_csv("E-commerce Customer Behavior - Sheet1.csv")
# Elimination of null values
df = df.dropna(axis= 0)

# A/B Test: Average Rating by Discount Applied
group_1 = df[df['Discount Applied'] == True]['Average Rating']
group_2 = df[df['Discount Applied'] == False]['Average Rating']

t_stat, p_value = stats.ttest_ind(group_1, group_2)
print(f'T-statistic: {t_stat}, P-value: {p_value}')

"""
    Results: 
        T-statistic: -1.43197218059624
        P-value: 0.15305451120901492

    With a P-value of 0.15 we cannot affirm that there are significant differences 
    between the group that received the Discount and the one that did not receive it 
    when evaluating the Average Rating.
"""

# A/B Test: Satisfaction Level by Discount Applied
contingency_table = pd.crosstab(df['Satisfaction Level'], df['Discount Applied'])
print(contingency_table)

chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)

print("Chi2 Stat:", chi2_stat)
print("P-value:", p_value)
print("Degrees of Freedom:", dof)
print("Expected Frequencies:")
print(expected)

"""
    Results:
        Chi2 Stat: 223.38788412881914              
        P-value: 3.1041114044899326e-49
        Degrees of Freedom: 2
        Expected Frequencies:  [[53.19252874 53.80747126]
                                [62.1408046  62.8591954 ]
                                [57.66666667 58.33333333]]
                                
    Based on the Chi-square test results, we can conclude that there is a 
    statistically significant relationship between receiving a discount and 
    customer satisfaction, with discounted customers tending to be more dissatisfied.
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Encoding categorical variables
label_encoder = LabelEncoder()

df['Satisfaction Level'] = label_encoder.fit_transform(df['Satisfaction Level'])

df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['City'] = label_encoder.fit_transform(df['City'])
df['Membership Type'] = label_encoder.fit_transform(df['Membership Type'])
df['Discount Applied'] = label_encoder.fit_transform(df['Discount Applied'])

# Training and testing sets.
X = df[['Age', 'Total Spend', 'Items Purchased', 'Discount Applied',
        'Days Since Last Purchase', 'Gender', 'City', 'Membership Type', 'Average Rating']]
y = df['Satisfaction Level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification report (precision, recall, F1-score)
print("Classification Report:")
print(classification_report(y_test, y_pred))

"""
    Results: 
        Accuracy: 0.9857
        Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.96      0.98        25
           1       0.96      1.00      0.98        26
           2       1.00      1.00      1.00        19

    accuracy                           0.99        70
   macro avg       0.99      0.99      0.99        70
weighted avg       0.99      0.99      0.99        70

The model demonstrates excellent performance with 99% accuracy and high precision, recall, 
and F1-scores across all satisfaction classes, effectively predicting customer satisfaction levels.

"""
# Graph of predictions and successes for each level of satisfaction
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()