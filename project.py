import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
import pickle

# Load the dataset
data = pd.read_csv('troop_movements.csv')

# Create EORR

emp_or_res = data.groupby('empire_or_resistance').size().reset_index(name='Count')
print(emp_or_res.head())

homeworld = data.groupby('homeworld')['homeworld'].count()
# print(homeworld.head())

unit_type = data.groupby('unit_type')['unit_type'].count()
# print(unit_type.head())

data['is_resistance'] = data['empire_or_resistance'] == 'resistance'
# print(data.head())

data['unit_type'] = data['unit_type'].apply(lambda x: "unknown" if x =='invalid_unit' else x)

sns.set_theme(style="whitegrid")

# bar Plot
# plt.figure(figsize=(10, 6))
# sns.barplot(x='empire_or_resistance', y='Count', data=emp_or_res)
# plt.title('Count By Empire vs Resistance')
# plt.xlabel('Empire vs Resistance')
# plt.ylabel('Count')
# plt.show()

dtm = DecisionTreeClassifier()

y = data['is_resistance']
X = pd.get_dummies(data[['homeworld', 'unit_type']], drop_first=True)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)




dtm.fit(X_train, y_train)

print("Training accuracy:", dtm.score(X_train, y_train))
print("Testing accuracy:", dtm.score(X_test, y_test))

importances = dtm.feature_importances_
features_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
})

# plt.figure(figsize=(10, 6))
# sns.barplot(x='Feature', y='Importance', data=features_df)
# plt.title('Feature Importances')
# plt.xlabel('Features')
# plt.ylabel('Importance')
# plt.show()


# Specify the file path to save the pipeline
file_path = 'emp_res_model.pkl'

# Save the pipeline to disk
with open(file_path, 'wb') as file:
    pickle.dump(dtm, file)