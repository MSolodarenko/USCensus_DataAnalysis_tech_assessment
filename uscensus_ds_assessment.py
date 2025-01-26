import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Step 0: Load and understand the data structure
# load the datasets without header
train_data = pd.read_csv('USCensusData/census_income_learn.csv', header=None)
test_data = pd.read_csv('USCensusData/census_income_test.csv', header=None)

# manually recheck names for each column using the head of train_data and metadata file
columns = ["age","class_of_worrer","industry_recode","occup_recode","education","wage_per_hour","enroll_in_edu_inst_last_wk","marital_stat","industry_code","occup_code","race","hispanic_origin","sex","is_labor_union","unemployment_reason","full_or_part_time_employment_stat","capital_gains","capital_losses","dividends_from_stocks","tax_filler_stat","region_of_prev_residence","state_of_prev_residence","detailed_household_and_family_stat","detailed_household_summary_in_household","instance_weight","migration_code_change_in_msa","migration_code_change_in_reg","migration_code_move_within_reg","live_in_house_1_year_ago","migration_prev_res_in_sunbelt","num_persons_worked_for_employer","family_members_under_18","country_of_birth_father","country_of_birth_mother","country_of_birth_self","citizenship","own_business_or_self_employed","fill_inc_questionnaire_for_veterans_admin","veterans_benefits","weeks_worked_in_year","year","income_class"]

# assign column names to the dataframes
train_data.columns = columns
test_data.columns = columns

# ensure the target variable (income_class) is binary and encoded properly
train_data['income_class'] = train_data['income_class'].apply(lambda x: 1 if x == " 50000+." else 0)
test_data['income_class'] = test_data['income_class'].apply(lambda x: 1 if x == " 50000+." else 0)

# # display basic information
print("Train data info:")
print(train_data.info())
print("Test data info:")
print(test_data.info())


# Step 1: Exploratory Data Analysis
# check for missing values
print("Number of missing values in each column:")
print(train_data.isnull().sum())

# summary statistics
print("Summary statictics")
print(train_data.describe().to_string())

# check the distribution of the income_class target variable
sns.countplot(x='income_class', data=train_data)
plt.title('Income class distribution (binary)')
plt.xticks(ticks=[0,1], labels=["<=50K",">50K"])
plt.xlabel('Income Class')
plt.ylabel('Count')
plt.show()

# Explore relationships between feature and income_class
# relationship between specific numerical feature and the income_class
numerical_features = train_data.select_dtypes(include=['int64', 'float64']).columns
# Define the grid dimensions
num_features = len(numerical_features)
rows = 2#int(np.floor(np.sqrt(num_features)))  # Number of rows in the grid
cols = (num_features + 1) // rows  # Calculate the required columns dynamically
# Create a grid of plots
fig, axes = plt.subplots(rows, cols, figsize=(14, 14))
axes = axes.flatten()  # Flatten the 2D array of axes for easier indexing
# Plot each feature in the grid
for i, column in enumerate(numerical_features):
    sns.boxplot(x='income_class', y=column, data=train_data, ax=axes[i])
    axes[i].set_title(f'{column} vs income_class')
    axes[i].set_xticks(ticks=[0, 1], labels=["<=50K", ">50K"])
    axes[i].set_xlabel('Income Class')
    axes[i].set_ylabel(column)
# Adjust layout
plt.tight_layout()
plt.show()

# correlation heatmap (numerical features only + income_class)
correlation_matrix = train_data[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlations with income_class')
plt.show()
exit()

numerical_features = train_data.select_dtypes(include=['int64', 'float64']).columns.drop(['income_class'])
categorical_features = train_data.select_dtypes(include=['object']).columns

# Step 2: Data Preparation
# no missing values
# but if there were missing values:
# train_data.fillna(method='ffill', inplace=True)
# test_data.fillna(method='ffill', inplace=True)

# encode categorical features
encoder = LabelEncoder()
for col in categorical_features:
    train_data[col] = encoder.fit_transform(train_data[col])
    test_data[col] = encoder.fit_transform(test_data[col])

# scale numerical features
scaler = StandardScaler()
train_data[numerical_features] = scaler.fit_transform(train_data[numerical_features])
test_data[numerical_features] = scaler.fit_transform(test_data[numerical_features])

# split datasets into predictors (X) and target (y) subsets
X_train = train_data.drop('income_class', axis=1)
y_train = train_data['income_class'].astype(int)
X_test = test_data.drop('income_class', axis=1)
y_test = test_data['income_class'].astype(int)

print("Train data info:")
print(X_train.info())
print(X_train.describe())
print(y_train.info())
print(y_train.describe())

# Feature importance using Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Plot feature importance
feature_importances = pd.DataFrame({'Feature': X_train.columns, 'Importance': rf.feature_importances_})
feature_importances.sort_values(by='Importance', ascending=False, inplace=True)

sns.barplot(x='Importance', y='Feature', data=feature_importances)
plt.title('Feature Importances')
plt.show()
exit()

# # Step 3: Data Modelling and Evaluation
# function for evaluating models
def evaluate_model(y_true, y_pred, model_name):
    print(f"{model_name} - Accuracy: {accuracy_score(y_true, y_pred):.2f}")
    print(classification_report(y_true, y_pred))

# # Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
evaluate_model(y_test, lr_pred, "Logistic Regression")

# # Random Forest
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
evaluate_model(y_test, rf_pred, "Random Forest")

# # XGBoost
xgb = XGBClassifier(random_state=42, eval_metric='logloss')
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)
evaluate_model(y_test, xgb_pred, "XGBoost")

# Step 2a: Adjustments
# feature engineering: log-transform skewed features
for col in ['capital_gains', 'capital_losses', 'dividends_from_stocks']:
    train_data[col] = np.log1p(train_data[col])
    test_data[col] = np.log1p(test_data[col])
# 
# Address class imbalance using SMOTE (there are significantly more examples of people with income below 50k)
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
y_train_balanced = y_train_balanced.astype(int)

# Step 3a: Hyperparameter Tuning with GridSearchCV
# Define parameter grids
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5]
}

# Random Forest adjusted
rf_adj = RandomForestClassifier(random_state=42)
rf_adj_grid = GridSearchCV(rf_adj, rf_param_grid, cv=3, scoring='f1', n_jobs=-1)
rf_adj_grid.fit(X_train_balanced, y_train_balanced)
# Random Forest adjusted Evaluation
rf_adj_pred = rf_adj_grid.best_estimator_.predict(X_test)
evaluate_model(y_test, rf_adj_pred, "Random Forest (data adj + balanced trainset + optimized w/ GridSearchCV)")

# XGBoost adjusted
xgb_adj = XGBClassifier(random_state=42, eval_metric='logloss')
xgb_adj.fit(X_train_balanced, y_train_balanced)
# XGBoost adjusted Evaluation
xgb_adj_pred = xgb_adj.predict(X_test)
evaluate_model(y_test, xgb_adj_pred, "XGBoost (data adj + balanced trainset)")

# Logistic Regression adjusted
lr_adj = LogisticRegression(max_iter=1000, class_weight='balanced')
lr_adj.fit(X_train_balanced, y_train_balanced)
# Logistic Regression adjusted Evaluation
lr_adj_pred = lr.predict(X_test)
evaluate_model(y_test, lr_adj_pred, "Logistic Regression (data adj + balanced trainset + 1000iters + balanced class weight)")

# Step 3b: Deploy Ensemble Model
# Get probabilities from each adjusted model
lr_probs = lr_adj.predict_proba(X_test)
rf_probs = rf_adj_grid.best_estimator_.predict_proba(X_test)
xgb_probs = xgb_adj.predict_proba(X_test)
# Average the probabilities (soft voting)
ensemble_probs = (lr_probs + rf_probs + xgb_probs) / 3
# Final prediction (class with the highest average probability)
ensemble_pred = np.argmax(ensemble_probs, axis=1)
# Evaluate the ensemble
evaluate_model(y_test, ensemble_pred, "Custom Soft Voting Ensemble")