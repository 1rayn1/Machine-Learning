import pandas as pd #used for reading and data manipulation
from sklearn.metrics import mean_absolute_error #used to measure model performance
from sklearn.model_selection import train_test_split #used to split data into training and validation sets
from sklearn.tree import DecisionTreeRegressor #used to create the model

def extraspace():
    print("\n")

'''
replacement_data_file = pd.DataFrame({"Name": ["John Doe", "Jane Smith", "Peter Jones"],
              "Age": [30, 25, 42],
              "City": ["New York", "Los Angeles", "Chicago"],
              "Salary": [100000, 250000, 4728390]})
print(replacement_data_file)

extraspace()
'''





'''
table_data = pd.DataFrame([[35,21],[41,34]], columns = ['Apple','Banana'], index = ['Store 1', 'Store 2'])
print(table_data.Apple)
extraspace()
print(table_data['Apple'])
extraspace()
print(table_data)
extraspace()
print(table_data['Apple']['Store 2'])
extraspace()
print(table_data.loc['Store 2']['Apple'])
extraspace()
print(table_data.iloc[0])
extraspace()
print(table_data.iloc[:,0])
extraspace()
print(table_data.iloc[:3,0])
extraspace()
print(table_data.iloc[[0,1],0])
extraspace()
print(table_data.iloc[-1:])
#extraspace()
#print(table_data.loc[0, 'Apple'])
#Note: The above line will give an error because 'loc' requires label names, not integer positions.
extraspace()
print(table_data.loc[:, 'Apple'])
extraspace()
#the 'loc' function is mainly used when we want to select rows and columns based on their labels.
#Whereas 'iloc' is used when choosing rows and columns based on specific positions.
print(table_data.set_index("Apple"))
extraspace()
print(table_data == "Apple")
extraspace()
print(table_data[table_data == "Apple"])
extraspace()
print(table_data.loc[(table_data['Apple'] > 36) & (table_data['Banana'] < 30)])
#The ampersand (&) is used for element-wise logical AND operations in pandas.
print(table_data.loc[(table_data['Apple'] > 36) | (table_data['Banana'] < 30)])
#Similarly, the pipe (|) is used for element-wise logical OR operations.
extraspace()
print(table_data.loc[table_data.Apple.isin(['Store 1', 'Store 2'])])
#The above line checks if the values in the 'Apple' column are in the list ['Store 1', 'Store 2']. (which they are not)
extraspace()   
print(table_data.loc[table_data.Banana.notnull()])
#The above line filters the DataFrame to include only rows where the 'Banana' column is not null.
extraspace()
table_data['Cherry'] = 'New Fruit'
print(table_data['Cherry'])
table_data['index_backwards'] = range(len(table_data), 0, -1)
print(table_data['index_backwards'])
extraspace()

'''









'''
quantities = ['4 cups', '1 cup', '2 large', '1 can']
items = ['flour', 'sugar', 'eggs', 'milk']
recipe = pd.Series(quantities, index = items, name = 'Cake Recipe')
print(recipe)
#note: if you add a ',' index_col = 0 to read_csv, it will use the first column as the index
extraspace()
'''










'''
example_data = pd.DataFrame({"Name": ["John Doe", "Jane Smith", "Peter Jones"],
              "Age": [30, 25, 42],
              "City": ["New York", "Los Angeles", "Chicago"],
              "Salary": [100000, 250000, 4728390]})
example_data.to_csv('NewExample.csv')
'''