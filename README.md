# Group C: Python Assessment


## Description
In this data science project, our team of five students will work collaboratively to load and analyze a selected World University Ranking dataset . We will leverage the power of Python libraries such as pandas, numpy, seaborn, and matplotlib to accomplish our objectives.

Data - https://github.com/Nosa-khare/python-assessment/tree/main/dataset


### Members
    Olaide oludare Wasiu | sS216546
    Nosakhare Edokpayi | S4214240
    Remsha Farooq | S4218275
    Uchechukwu Osita Ikwu | S4216861
    Yusuf Segun Ajibade | S4216782



## Table of Contents


- [Installation](#installation)
- [Dataset](#data)
- [Data Integration](#data-integration)
  - [Import csv files](#import-csv-files)
  - [Merge dataframes](#merge-dataframes)
- [Data Cleaning](#data-cleaning)  
  - [Rename columns](#rename-columns)
  - [Rearrange column positions](#rearrange-column-positions)
  - [Missing Values](#missing-values)
    - [Categorize the missing data by the 'year' column](##categorize-the-missing-data-by-the-year-column)
    - [Clean missing data](##clean-missing-data)
  - [Duplicates](#duplicates)
- [Data Exploration](#data-exploration)
  - [Correlation Analysis](#correlation-analysis)
  - [Research Question 1](#research-question-1)
  - [Research Question 2](#research-question-2)
    - [Hypothesis 1: Universities with higher publication counts have a greater influence and citations](#hypothesis-1-universities-with-higher-publication-counts-have-a-greater-influence-and-citations)
    - [Hypothesis 2: Universities with higher quality of education have higher alumni employment rates](#hypothesis-2-universities-with-higher-quality-of-education-have-higher-alumni-employment-rates)
    - [Hypothesis 3: Universities with a higher number of patents also have higher scores](#hypothesis-3-universities-with-a-higher-number-of-patents-also-have-higher-scores)
  - [Research Question 3](#research-question-3)
  - [Research Question 4](#research-question-4)
  - [Research Question 5](#research-question-5)
  - [Research Question 6](#research-question-6)



## Installation
To run this project, please ensure that Python is installed on your machine. You can install Python by downloading it from the official Python website (https://www.python.org/) or by using the Anaconda distribution, which provides a comprehensive Python environment with many useful packages pre-installed.

Please make sure to install a compatible version of Python (version 3.9 and above). 

Once Python is installed, you can proceed with setting up the project environment and installing any necessary dependencies as mentioned in the project instructions.


## Dependencies
The following dependencies are required to run the project:

matplotlib version: 3.5.2
numpy version: 1.21.5
pandas version: 1.4.4
plotly version: 5.9.0
scipy version: 1.9.1
seaborn version: 0.11.2
scikit-learn version: 1.0.2

You can install these dependencies by running the following command:
```pip install matplotlib numpy pandas plotly scipy seaborn scikit-learn```


## Data
For this project, we opted to utilize a global University ranking dataset sourced from The Center for World University Rankings (CWUR). The dataset spans the years 2012 to 2017 and encompasses 4,200 entries with 14 columns, encompassing information on world rankings, quality, publications, and scores of various institutions. 

This dataset offers an opportunity for in-depth data analysis, enabling us to uncover valuable insights and visualize trends that can help inform universities efforts to improve educational and research standards.Initially, we utilized a pre-cleaned version of the dataset available on Kaggle, but upon realizing the need for an update, we conducted web scraping directly from the CWUR main website. This update was necessary due to changes in variables collected stemming from the organization's updated methodology.


download data files here: https://github.com/Nosa-khare/python-assessment/tree/main/dataset


## Data Integration

### Get CSV Files
The purpose of this code is to retrieve a list of CSV files from a specified directory.

```python
def get_csv_files(csv_path):
    files = os.listdir(csv_path)
    csv_files = [file for file in files if file.endswith('.csv')]
    return csv_files

dataset_dir = os.getcwd() + '\\dataset'
csv_files = get_csv_files(dataset_dir)
print(csv_files)
print(dataset_dir)
```

This code defines a function called get_csv_files() that takes a single argument csv_path. It uses the os module to retrieve a list of all files in the specified directory. It then filters the list to include only files with the .csv extension. The function returns the list of CSV files.

To use this function, construct the csv_path variable with the path to the directory containing the CSV files. Then, call the get_csv_files() function with the csv_path variable and assign the returned list of CSV files to the csv_files variable. Finally, print the list of CSV files and the dataset_dir variable.

**Note:** Make sure the '/dataset' directory exists in the current working directory and contains the required CSV files (see Data section for download link).


### Import csv files

The purpose of this code is to read a CSV file, extract the year from its file name, and add a new 'year' column to the DataFrame.

```python
def import_with_year_column(csv_file):
    file_basename = os.path.basename(csv_file)
    year = file_basename[9:13]
    df = pd.read_csv(csv_file)
    df['year'] = year
    return df
```
This code defines a function called import_with_year_column() that takes a CSV file path as input and returns a Pandas DataFrame.

It extracts the base name of the CSV file from the given csv_file path. Then, it extracts the year from the file name by slicing the file_basename string assuming the year is present in positions 9 to 12, and reads the CSV file using Pandas' read_csv() function.

A new column called 'year' is added to the DataFrame df and assigns it the value of the year variable. 

the function then returns the modified DataFrame.

To use this function, call import_with_year_column() with the path to the desired CSV file as the argument. The function will return a DataFrame with an additional 'year' column based on the file name.


### Merge dataframes

The merge_df_rows() merges rows from multiple CSV files into a single dataframe. It checks that the CSV files have the same column structure and contain a "year" column.

#### Usage:
```python
csv_files_path = [dataset_dir + '\\' + csv_file for csv_file in csv_files]
df = merge_df_rows(csv_files_path)
df_merged = df
```

#### Function Signature:
def merge_df_rows(csv_paths)

The function takes a list of CSV file paths as input and compares the column names of subsequent CSV files with the first CSV file in the list, which serves as the reference dataframe. If the column names match, the rows are concatenated using pd.concat to create the merged dataframe. If the column names are not compatible, the CSV file is added to a list of incompatible files.

After processing all CSV files, the function checks if any files were not compatible. If there are incompatible files, it prints a message indicating the files that are not compatible. Otherwise, it prints a success message.

The merged dataframe is returned as the output of the function. In the provided code snippet, it is assigned to the variable df_merged for further usage



## Data Cleaning


### get column names

```python
df.columns.to_list() 
```

This retrieves the column names of a DataFrame df and converts them into a Python list.

The columns attribute of a DataFrame contains the column labels or names. By calling the to_list() method on the columns attribute, the column names are extracted and converted into a list.


# rename column names
 a function rename_columns that renames the columns of a DataFrame based on a provided dictionary. 
 
 ```python
 def rename_columns(df, colname_dict):
    df = df.rename(columns=colname_dict)
    return df
```

 It also includes additional steps to standardize the column names by replacing whitespace characters with underscores and converting them to lowercase.

The rename_columns function, easily renames the columns passing the DataFrame and a dictionary mapping the current column names to the desired new names. The function modifies the DataFrame in place and returns it.


### Rearrange column positions

The code snippet creates a new DataFrame df_sorted_cols by selecting specific columns from the df_renamed DataFrame. It then displays information about the DataFrame using the info() method and shows the first few rows using the head() method.

```python
df_renamed.iloc[:, [13, 1, 2, 0, *range(3,13)]]
```

code selects columns from df_renamed based on their indices. It selects the 14th column (index 13), 2nd column (index 1), 3rd column (index 2), 1st column (index 0), and columns 4 to 13 (indices 3 to 12) in that order. This rearrangement of columns is stored in the df_sorted_cols DataFrame.


### Missing Values

The code snippet assigns the df_sorted_cols DataFrame to the variable df and then prints the count of missing values in each column using the isna().sum() method.

```python
df_null_rows = df[df.isnull().any(axis=1)]
```
creates a new DataFrame df_null_rows by selecting rows from df that contain at least one missing value. It uses the isnull().any(axis=1) condition to check for any missing values across the columns (axis=1).
Finally, the code displays the df_null_rows DataFrame, which contains the rows with missing values.


#### Categorize the missing data by the 'year' column

The code categorizes the missing data from the df_null_rows DataFrame based on the 'year' column. It retrieves the unique values of the 'year' column in the missing data using the unique() method and converts the result to a list using the tolist() method.

inspect data of each missing yeaR
This code assigns the filtered data to the variable data_2012 by filtering df based on the condition df['year'] == '2012'. It then uses data_2012.isna().sum() to calculate the missing value count for each column in data_2012. Finally, it prints the result with a formatted string that includes the year in the output.


#### Clean missing data

Filtering and updating the "broad_impact" column for the years 2012 and 2013 based on specific conditions and values from other columns.

```python
institutions_2012 = df[df['year'] == '2012']['institution'].tolist():
```
Filters the DataFrame to select rows where the 'year' column is '2012' and extracts the values from the 'institution' column, storing them in the institutions_2012 list.

For each institution in institutions_2012, it:

Retrieves the row(s) in 2014 for the current institution: 
```python
row_2014 = df[(df['institution'] == institution) & (df['year'] == '2014')]
```
If no row exists for the institution in 2014, sets the 'broad_impact' rank to 0.

If a row exists, retrieves the 'broad_impact' value at index 0 from row_2014 and assigns it to the 'broad_impact' column of the corresponding rows in 2012: 
```python
df.loc[(df['institution'] == institution) & (df['year'] == '2012'), 'broad_impact'] = broad_impact_rank.
```

Similar process is performed for the year 2013 using the institution_2013 list:

Retrieves the row(s) in 2014 and 2015 for each institution: 
```python
row_2014 = df[(df['institution'] == institution) & (df['year'] == '2014')] and row_2015 = df[(df['institution'] == institution) & (df['year'] == '2015')]
```

If no rows exist for the institution in both 2014 and 2015, sets the 'broad_impact' rank to 0.

If rows exist, calculates the mean of the 'broad_impact' values at index 0 from row_2014 and row_2015 and assigns it to the 'broad_impact' column of the corresponding rows in 2013.


### Duplicates

```python
years = df['year'].unique().tolist()
```
The code retrieves the unique years in the 'year' column and stores them in the years list.

An empty DataFrame called duplicate_rows is initialized to store the duplicate rows.

```python
# iterate over each year:
for year in years:

    # check for duplicate in current year and assign to duplicates
    duplicates = df[df['year'] == year].duplicated(subset=['institution'])
```
For each year, the code checks for duplicates in the 'institution' column within that year using the duplicated() function.

```python
 #  subset the main DataFrame by the duplicates and append to the duplicate_rows DataFrame
 duplicate_rows = pd.concat([duplicate_rows, df[(df['year'] == year) & duplicates]])
```
The rows with duplicates are appended to the duplicate_rows DataFrame using pd.concat().

Finally, the code prints the duplicate_rows DataFrame.




## Data Exploration


### Correlation Analysis

This code computes the correlation matrix using the corr() function on the DataFrame df. The resulting correlation matrix is stored in the corr_matrix variable.

```python
# Compute the correlation matrix
corr_matrix = df.corr()
corr_matrix
```
This code computes the correlation matrix using the corr() function on the DataFrame df. The resulting correlation matrix is stored in the corr_matrix variable.


```python
# Create a heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap='coolwarm', annot=True, fmt=".2f", vmin=-1, vmax=1)

plt.title('Correlation Heatmap', fontsize=12, fontweight='bold')

# Adjust the xtick rotation
plt.xticks(rotation=45)

plt.show()
```
This then creates a heatmap using the seaborn library. The heatmap visualizes the correlation matrix, with higher values indicated by warmer colors and lower values indicated by cooler colors. 
The ```python cmap='coolwarm'``` parameter sets the color palette for the heatmap. The ```python annot=True``` parameter displays the correlation values in each cell. 
```python The fmt=".2f" ``` parameter formats the values to two decimal places. The ```python vmin and vmax ``` parameters set the range of values for the color scale. The title and x-axis tick labels are adjusted for better readability.


### Strong correlation

This section calculates strong correlations among variables.  

```python 
# Set a threshold for strong correlation
strong_correlation_threshold = 0.8
```
It sets a threshold for strong correlation and finds the variables that have correlations above this threshold.

```python
# Find variables with strong correlations
strong_correlation_columns = {}

for i in range(len(corr_matrix.columns)):
    correlations = corr_matrix_abs.iloc[i, :]
    strong_correlations = correlations[correlations >= strong_correlation_threshold]
    if len(strong_correlations) > 1:
        strong_correlation_columns[corr_matrix.columns[i]] = correlations
```
This code iterates over the columns of the correlation matrix and finds the variables (columns) that have strong correlations with other variables. It creates a dictionary strong_correlation_columns where the keys are the variable names with strong correlations, and the values are the corresponding correlation values.

#### heatmap
it creates a heatmap using the heatmap() function from the seaborn library. The heatmap visualizes the correlation values, where higher values are represented by warmer colors.

```python
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=True, fmt=".2f", vmin=-1, vmax=1)
```
The cmap='coolwarm' argument sets the color map, annot=True displays the correlation values on the heatmap, fmt=".2f" formats the displayed values as floating-point numbers with two decimal places, and vmin and vmax set the range of values for the color map. 

Finally, the title() function sets the title of the plot, and xticks(rotation=45) adjusts the rotation of the x-axis tick labels.

#### network graph 
This code segment visualizes the strong correlations as a network graph. It uses the networkx library to create a graph (G). 
```python
# Add edges to the graph based on the strong correlations
for variable1 in strong_correlation_columns:
    for variable2 in strong_correlation_columns[variable1].index:
        G.add_edge(variable1, variable2)
```
Then, it adds edges to the graph based on the strong correlations found earlier. 

The layout of the graph is set using spring_layout, which positions the nodes using the Fruchterman-Reingold force-directed algorithm. 

```python
# Draw the nodes and edges
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
nx.draw_networkx_edges(G, pos, edge_color='gray')
```
The draw_networkx_nodes and draw_networkx_edges functions are used to draw the nodes and edges of the graph. Labels are added to the nodes using draw_networkx_labels. 

The title() function sets the title of the plot, and axis('off') removes the axis from the plot. Finally, the graph is displayed using plt.show().

Overall, this code segment includes visualization techniques to represent the correlations: a heatmap and a network graph.


### Research Question 1

This code segment focuses on exploring the total number of institutions by country in the dataset and analyzing the top 5 countries in more detail.


```python
df_uc = pd.read_csv('cwur_cleaned.csv')
```
The line loads the dataset from a CSV file into a DataFrame called df_uc. The subsequent df = df_uc line assigns the DataFrame to df for further processing.

```python
def sum_institutions_by_country(df):
    country_totals = df.groupby('country')['institution'].nunique().reset_index()
    country_totals.columns = ['country', 'total_institutions']
    country_totals['total_institutions'] = df.groupby('country')['institution'].nunique().values
    
    return country_totals
```
The sum_institutions_by_country() function calculates the total number of institutions by country. It uses the groupby() function to group the data by country and then counts the unique institutions within each group using nunique(). The function returns a DataFrame country_totals with two columns: 'country' and 'total_institutions'.

The country_aggregate variable stores the result of calling sum_institutions_by_country(df), which computes the total number of institutions by country using the provided DataFrame df.

The sorted_country_aggregate variable sorts the country_aggregate DataFrame by the 'total_institutions' column in descending order, giving a sorted representation of countries based on the number of institutions.

The top_20_countries variable selects the top 20 countries with the highest institution counts from the sorted_country_aggregate DataFrame.

```python
sns.barplot(data=top_20_countries, 
            x='country', y='total_institutions', 
            color='royalblue', saturation=0.7, ci=None)
```
A bar chart is created using sns.barplot() to visualize the total number of institutions by country for the top 20 countries. The 'country' column is plotted on the x-axis, 'total_institutions' is plotted on the y-axis, and the bars are colored in 'royalblue'. The chart is displayed using plt.show().

The code then focuses on the top 5 countries, stored in the top_5_countries variable. For each country, it creates a group DataFrame containing only the data for that country. The country name, the group DataFrame, and statistical analysis (using describe()) are printed for each country.


### Research Question 2

This code segment investigates how different factors contribute to a university's overall ranking.

For each HYPOTHESIS, the correlation matrix is calculated for the variables 'publications', 'influence', and 'citations'. The correlation matrix is then visualized as a heatmap using sns.heatmap(). The resulting correlation coefficients and p-values are printed.


### Research Question 3

This code segment analyzes the performance of universities from different countries in terms of overall ranking and individual factors.

First, the code calculates the average university ranking by country using the groupby function and the mean aggregation. The results are sorted and displayed in a bar plot.

```python
factors = ['education_quality', 'alumni_employment', 'faculty_quality', 'publications', 'influence', 'citations']

# Iterate through each factor and plot the average scores by country
for factor in factors:
    factor_ranking = df.groupby('country')[factor].mean().reset_index()
    sorted_factor_ranking = factor_ranking.sort_values(factor)
    top_10 = sorted_factor_ranking.head(10)
    print(top_10)
    print()


    plt.figure(figsize=(12, 6))
    sns.barplot(x=factor, y='country', data=top_10, color='royalblue' )
    plt.xlabel(f'Average Overall Rankings', fontsize=12)
    plt.ylabel('Country', fontsize=11)

    plt.title(f'Average {factor.capitalize()} Rankings by Country', fontsize=14)
    plt.yticks(fontsize=12)

    plt.show()

```
Next, the code analyzes individual factors such as education quality, alumni employment, faculty quality, publications, influence, and citations. For each factor, the average scores by country are calculated using the groupby function and visualized in separate bar plots.

```python
# Calculate the yearly average ranking per institution
yearly_avg_ranking = df.groupby(['year', 'institution'])['world_rank'].mean().reset_index()

top_50_rankings = yearly_avg_ranking.sort_values('world_rank').head(50)

# Define a formatting function
def format_rank(rank):
    return int(rank)

# Apply the formatting function to the 'world_rank' column
top_50_rankings['world_rank'] = top_50_rankings['world_rank'].apply(format_rank)

print(top_50_rankings)

```
Lastly, the code focuses on the ranking trend for individual institutions over the years. The yearly average ranking per institution is calculated, and the top 50 rankings are selected. The code then plots the average ranking trend for each institution using line plots, with each institution

```python
plt.gca().invert_yaxis()
```
Inverts the y-axis scale so that lower rank values (higher performance) appear at the top of the graph.

plt.show(): Displays the plot.


### Research Question 4

This code segment calculates the correlation matrix for the variables in the dataset and creates a heatmap visualization using seaborn's heatmap function. The correlation matrix measures the relationships between different variables, including the number of students and other factors. The heatmap provides a color-coded representation of these correlations.


### Research Question 5

This segment of code aims to explore and compare the performance, ranking, and resources of public and private universities.

```python
# Create a new column indicating university type (public or private)
df["university_type"] = np.where(df["country"] == "USA", "Private", "Public")
print(df["university_type"])
```
It creates a new column in the DataFrame df called "university_type" based on the condition that if the country is "USA," the university type is set as "Private," otherwise it is set as "Public." The np.where function is used to apply this condition.


### Research Question 6

This code segment aims to predict a university's future performance or ranking using a decision tree algorithm. It splits it into training and testing sets.

```python
predictions = model.predict(x_test)
```
The code applies predictions on the test data (x_test) using the trained model (model.predict).

```python
score = accuracy_score(y_test, predictions)
```

The accuracy of the predictions is calculated using the accuracy_score function from scikit-learn, comparing the predicted values (predictions) with the actual values (y_test).

This provides a way to assess the model's performance in predicting university rankings or performance. The variable "predictions" contains the predicted values for the test data.


## Conclusion

Summarize the key findings and insights from your data science project.

Feel free to adjust the sections and their descriptions according to your specific project needs. Remember to provide clear explanations, code instructions, and any necessary dependencies or assumptions for running the code successfully.
