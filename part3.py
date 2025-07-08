import pandas as pd
import pickle

df = pd.read_csv('troop_movements_1m.csv')

df['unit_type'] = df['unit_type'].apply(lambda x: "unknown" if x =='invalid_unit' else x)

unit_type = df.groupby('unit_type')['unit_type'].count()
#print(unit_type.head())

df['location_x'] = df['location_x'].ffill()
df['location_y'] = df['location_y'].ffill()
location = df.groupby('location_x')['location_x'].count()
#print(location.head())

df.to_parquet('troop_movements_1m.parquet')

# Specify the file path to save the pipeline
file_path = 'emp_res_model.pkl'

# Save the pipeline to disk
with open(file_path, 'rb') as file:
    model = pickle.load(file)

df_p = pd.read_parquet('troop_movements_1m.parquet')
X = pd.get_dummies(df_p[['homeworld', 'unit_type']], drop_first=True)

predictions = model.predict(X)
print(predictions)
df_p['predictions'] = predictions
print(df_p)