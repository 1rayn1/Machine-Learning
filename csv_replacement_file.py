import pandas as pd #used for reading and data manipulation
from sklearn.metrics import mean_absolute_error #used to measure model performance
from sklearn.model_selection import train_test_split #used to split data into training and validation sets
from sklearn.tree import DecisionTreeRegressor #used to create the model

replacement_data_file = pd.DataFrame({"Name": ["John Doe", "Jane Smith", "Peter Jones"],
              "Age": [30, 25, 42],
              "City": ["New York", "Los Angeles", "Chicago"],
              "Salary": [100000, 250000, 4728390]})

table_data = pd.DataFrame([[35,21],[41,34]], columns = ['Apple','Banana'], index = ['Store 1', 'Store 2'])

quantities = ['4 cups', '1 cup', '2 large', '1 can']
items = ['flour', 'sugar', 'eggs', 'milk']
recipe = pd.Series(quantities, index = items, name = 'Cake Recipe')
#note: if you add a , index_col = 0 to read_csv, it will use the first column as the index
example_data = pd.DataFrame({"Name": ["John Doe", "Jane Smith", "Peter Jones"],
              "Age": [30, 25, 42],
              "City": ["New York", "Los Angeles", "Chicago"],
              "Salary": [100000, 250000, 4728390]})

example_data.to_csv('NewExample.csv')

print(replacement_data_file)
print(table_data)
print(recipe)