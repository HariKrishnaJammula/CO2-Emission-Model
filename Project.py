import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, f1_score, classification_report

st.title("Predicting Banking Term Deposit Subscription for Marketing Strategy Optimization")



uploaded_file = st.file_uploader("Upload your input CSV file", type=["csv"])





# Check if a file has been uploaded
if uploaded_file is not None:
    # Read the file
    data = pd.read_csv(uploaded_file)

    # Show the first 5 rows of the data
    st.write("First 5 rows of the data:")
    st.write(data.head())

    # Show the shape of the data
    st.write("Shape of the data:")
    st.write(data.shape)

    # Show the data types of the columns
    st.write("Data types of the columns:")
    st.write(data.dtypes)

    # Show the summary statistics of the data
    st.write("Summary statistics of the data:")
    st.write(data.describe())

    # Show the missing values in the data
    st.write("Missing values in the data:")
    st.write(data.isnull().sum())


    # Detect and display duplicate rows
    duplicate_rows = data[data.duplicated()]
    st.write("Duplicate rows in the data:")
    st.write(duplicate_rows)


    # Histogram for numerical features
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader('Histogram for Numerical Features')
    numerical_features = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'day']
    for feature in numerical_features:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[feature], kde=True)
        st.pyplot()

    # Scatter Plot for numerical features
    st.subheader('Scatter Plot for Numerical Features')
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='age', y='balance', data=data)
    st.pyplot()

    

    data['deposit'] = data['deposit'].map({'yes': 1, 'no': 0})

    # Display the updated data
    st.subheader('Updated Data with "deposit" Feature Converted to 0s and 1s')
    st.write(data)

    # Create a bar graph and Pie chart for the 'deposit' feature
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.subheader('Bar Graph for "deposit" Feature')
    deposit_counts = data['deposit'].value_counts()
    plt.bar(deposit_counts.index, deposit_counts.values, color=['blue', 'green'])
    plt.xlabel('Deposit')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['No', 'Yes'])
    st.pyplot()

    deposit_counts = data['deposit'].value_counts()
    labels = deposit_counts.index
    sizes = deposit_counts.values
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.legend(labels=['0 - No', '1 - Yes'], loc='best')
    ax1.axis('equal')
    st.pyplot()



    # Create and display bar graphs for different feature comparisons
    features_to_compare = ['job', 'marital', 'education', 'default', 'housing']  # Add more features as needed

    for feature in features_to_compare:
        st.subheader(f'Bar Graph for "{feature}" and "Deposit" Features')
        plt.figure(figsize=(12, 8))
        sns.countplot(x=feature, hue='deposit', data=data)
        plt.title(f'{feature} vs. Deposit')
        plt.xlabel(feature.capitalize())
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.legend(title='Deposit', loc='upper right')
        st.pyplot()
   
    # Correlation Heatmap
    st.subheader('Correlation Heatmap')
    correlation_matrix = data[['age', 'balance', 'duration', 'campaign', 'pdays', 'previous','deposit']].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
    st.pyplot()

    # Create and display box plots for the selected numerical features
    st.subheader('Box Plots')
    col = ['age', 'balance', 'duration', 'campaign', 'pdays', 'previous', 'day']
    plt.figure(figsize=(15, 18))
    for i, v in enumerate(col):
        plt.subplot(4, 2, i+1)
        sns.boxplot(x=v, data=data, color='green')
        plt.title(f'Boxplot of {v}', size=20, color='red')
        plt.xlabel(f'{v}', size=15)
    plt.tight_layout()
    st.pyplot(plt)

    st.write('No missing values, no duplicates but we found some outliers in the features age balance duration pdays and previous')
    
    
    
    # Dropping 'default' and 'pdays' columns
    data.drop(columns=['default', 'pdays'], inplace=True)

    # Handling outliers for 'age' and 'campaign'
    age_outliers = data[(data['age'] < 18) | (data['age'] > 95)]
    data = data[(data['age'] >= 18) & (data['age'] <= 95)]

    campaign_outliers = data[data['campaign'] >= 32]
    data = data[data['campaign'] < 32]

    # Removing outliers for 'previous'
    previous_outliers = data[data['previous'] >= 31]
    data = data[data['previous'] < 31]
    st.write('so dropping default as no imp role most of its values are no similarly pdays also has no significance most of its values are -1 handling outliers these can be ignored and values lies b/w 18 - 95 need not to be remoevd as blaance gets higher ,client show interest on deposit should not be removed as with increase in duration the client shows more interest in deposit removing outliers')



    
    st.write(data.head())

    data = pd.get_dummies(data, columns=["job", "marital","education","housing","loan","contact","month","poutcome" ])
    st.write(data.head())

    # Split the data into features and target variable
    X = data.drop('deposit', axis=1)
    y = data['deposit']

# Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the Decision Tree model
    dc = DecisionTreeClassifier()
    dc.fit(X_train, y_train)

# Make predictions using the Decision Tree model
    dc_pred = dc.predict(X_test)

# Evaluate the Decision Tree model
    accuracy = accuracy_score(y_test, dc_pred)
    class_report = classification_report(y_test, dc_pred)

# Display the evaluation metrics
    st.subheader('Model Evaluation')
    st.subheader('Decision tree Model Evaluation Metrics')
    st.write(f'Accuracy: {accuracy}')
    st.write('Classification Report:')
    st.text(class_report)


    feature_importance = pd.Series(dc.feature_importances_, index=X.columns)
    feature_importance = feature_importance.sort_values(ascending=False)

    # Create a bar plot for feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importance, y=feature_importance.index, palette='viridis')
    plt.title('Decision Tree Feature Importance')
    plt.xlabel('Feature Importance')
    plt.ylabel('Features')
    st.pyplot(plt)

# Initialize and train the Random Forest model
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

# Make predictions using the Random Forest model
    rf_pred = rf.predict(X_test)

# Evaluate the Random Forest model
    rf_accuracy = accuracy_score(y_test, rf_pred)
    rf_class_report = classification_report(y_test, rf_pred)

# Display the evaluation metrics
    st.subheader('Random Forest Model Evaluation Metrics')
    st.write(f'Accuracy: {rf_accuracy}')
    st.write('Classification Report:')
    st.text(rf_class_report)

    # Feature Importance
    feature_importance = rf.feature_importances_

    # Create a DataFrame for better visualization
    feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importance})

    # Sort the features by importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot the feature importance bar graph
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df, palette='viridis')
    plt.title('Random Forest Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    st.pyplot(plt)
    st.write('The feature importance of a machine learning model, such as the Random Forest model, is a measure of the contribution of each feature in making accurate predictions. It helps you understand which features are more influential in the models decision-making process. Feature importance is often expressed as a percentage or a score')

   
