# E-commerce Customer Behavior Analysis

## Overview:
This project performs an analysis of customer behavior for an e-commerce company. It includes data exploration, statistical testing (A/B testing), and predictive modeling. The goal is to explore how various factors, such as discounts and customer satisfaction, impact sales behavior, and to develop a predictive model to classify customer satisfaction levels based on various features.

## Data:
The data used in this project is provided in a CSV file titled "E-commerce Customer Behavior - Sheet1.csv." (https://www.kaggle.com/datasets/uom190346a/e-commerce-customer-behavior-dataset) The dataset includes customer demographic information, purchase details, and customer satisfaction levels.

## Key Steps:

### 1. Data Exploration and Visualization:
- **Descriptive Statistics**: Basic statistics and information about the dataset, including column names, data types, and null values.
- **Data Cleaning**: Removal of null values from the dataset.
- **Categorical Variables Analysis**: A bar chart showing the distribution of categorical features such as `Gender`, `City`, `Membership Type`, etc.
- **Numerical Variables Analysis**: Box plots for visualizing the spread and outliers in numerical variables such as `Age`, `Total Spend`, `Items Purchased`, etc.
- **Total Spend by Category**: Box plots to explore the relationship between `Total Spend` and categorical variables.
- **Correlation Matrix**: A heatmap to visualize correlations between numerical variables.
- **Total Spend vs. Days Since Last Purchase**: A scatter plot with a regression line to explore the relationship between these two variables.

### 2. A/B Testing:
- **Average Rating by Discount Applied**: An independent T-test is performed to determine if there's a significant difference in the average rating between customers who received a discount and those who did not.
- **Satisfaction Level by Discount Applied**: A Chi-square test is conducted to assess whether there is a significant relationship between receiving a discount and customer satisfaction. The results indicate that discounts tend to be linked with lower satisfaction levels.

### 3. Predictive Modeling:
- **Customer Satisfaction Prediction**: A Random Forest Classifier is trained to predict customer satisfaction levels based on features like `Age`, `Total Spend`, `Items Purchased`, and more. The model achieves an accuracy of 99% and demonstrates high precision and recall for each satisfaction level.

### 4. Confusion Matrix: 
A heatmap of the confusion matrix is generated to visualize the model's classification performance for each satisfaction level.

## Libraries Used:
- **pandas**: For data manipulation and analysis.
- **numpy**: For numerical operations.
- **matplotlib** and **seaborn**: For data visualization.
- **scipy**: For statistical tests (e.g., T-tests and Chi-square tests).
- **scikit-learn**: For machine learning tasks such as model training, evaluation, and preprocessing.

## Results:
- The A/B testing results indicate that receiving a discount does not significantly impact the average rating (P-value > 0.05).
- The Chi-square test shows that there is a statistically significant relationship between receiving a discount and customer satisfaction, with discounted customers tending to be more dissatisfied.
- The predictive model (Random Forest Classifier) successfully predicts customer satisfaction levels with 99% accuracy, demonstrating the power of various features like `Age`, `Total Spend`, and `Days Since Last Purchase`.

## How to Use:
1. Clone this repository to your local machine.
2. Ensure you have the required Python libraries installed (pandas, numpy, matplotlib, seaborn, scikit-learn, scipy).
3. Load the data by placing the CSV file in the same directory or updating the file path in the script.
4. Run the Python scripts to perform data exploration, statistical testing, and modeling.

## Future Improvements:
- Test other classification models such as Logistic Regression or Support Vector Machines (SVM) for comparison.
- Explore other features such as prior knowledge of the brand and the product that might influence customer behavior.
- Conduct deeper segmentation analysis based on customer profiles.

## License:
This project is licensed under the MIT License - see the LICENSE file for details.
