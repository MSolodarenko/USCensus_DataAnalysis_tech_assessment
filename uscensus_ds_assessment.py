import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

# display basic information
print("Train data info:")
# print(train_data.info())
# print("Test data info:")
# print(test_data.info())


# Step 1: Exploratory Data Analysis
# check for missing values
print("Number of missing values in each column:")
# print(train_data.isnull().sum())

# summary statistics
print("Summary statictics")
# print(train_data.describe().to_string())
# print(train_data.describe())

# check the distribution of the income_class target variable
sns.countplot(x='income_class', data=train_data)
plt.title('Income class distribution (binary)')
plt.xticks(ticks=[0,1], labels=["<=50K",">50K"])
plt.xlabel('Income Class')
plt.ylabel('Count')
# plt.show()

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
# plt.show()

# correlation heatmap (numerical features only + income_class)
correlation_matrix = train_data[numerical_features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlations with income_class')
# plt.show()

