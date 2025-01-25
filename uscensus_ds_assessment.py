import pandas as pd

# Step 0: load and understand the data structure
# load the datasets without header
train_data = pd.read_csv('USCensusData/census_income_learn.csv', header=None)
test_data = pd.read_csv('USCensusData/census_income_test.csv', header=None)

# Manually recheck names for each column using the head of train_data and metadata file
columns = ["age","class_of_worrer","industry_recode","occup_recode","education","wage_per_hour","enroll_in_edu_inst_last_wk","marital_stat","industry_code","occup_code","race","hispanic_origin","sex","is_labor_union","unemployment_reason","full_or_part_time_employment_stat","capital_gains","capital_losses","dividends_from_stocks","tax_filler_stat","region_of_prev_residence","state_of_prev_residence","detailed_household_and_family_stat","detailed_household_summary_in_household","instance_weight","migration_code_change_in_msa","migration_code_change_in_reg","migration_code_move_within_reg","live_in_house_1_year_ago","migration_prev_res_in_sunbelt","num_persons_worked_for_employer","family_members_under_18","country_of_birth_father","country_of_birth_mother","country_of_birth_self","citizenship","own_business_or_self_employed","fill_inc_questionnaire_for_veterans_admin","veterans_benefits","weeks_worked_in_year","year","predict_income_binary"]

# assign column names to the dataframes
train_data.columns = columns
test_data.columns = columns

# display basic information
print("Train data info:")
print(train_data.info())
print("Test data info:")
print(test_data.info())

