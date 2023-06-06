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
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact Information](#contact-information)

- [Data Integration](#data-integration)
  - [Import csv files](#import-csv-files)
  - [Merge dataframes](#merge-dataframes)
- [Data Cleaning](#data-cleaning)
  - [Rename columns](#rename-columns)
  - [Rearrange column positions](#rearrange-column-positions)
  - [Missing Values](#missing-values)
    - [Categorize the missing data by the 'year' column](#categorize-the-missing-data-by-the-year-column)
    - [Clean missing data](#clean-missing-data)
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
This code defines a function called import_with_year_column() that takes a CSV file path as input and returns a Pandas DataFrame. It performs the following steps:

Extracts the base name of the CSV file from the given csv_file path.
Extracts the year from the file name by slicing the file_basename string assuming the year is present in positions 9 to 12 (inclusive).
Reads the CSV file using Pandas' read_csv() function.
Adds a new column called 'year' to the DataFrame df and assigns it the value of the year variable.
Returns the modified DataFrame.
To use this function, call import_with_year_column() with the path to the desired CSV file as the argument. The function will return a DataFrame with an additional 'year' column based on the file name.


### Merge dataframes

The merge_df_rows() merges rows from multiple CSV files into a single dataframe. It checks that the CSV files have the same column structure and contain a "year" column.

#### Usage:
csv_files_path = [dataset_dir + '\\' + csv_file for csv_file in csv_files]
df = merge_df_rows(csv_files_path)
df_merged = df

#### Function Signature:
def merge_df_rows(csv_paths)

The function takes a list of CSV file paths as input and compares the column names of subsequent CSV files with the first CSV file in the list, which serves as the reference dataframe. If the column names match, the rows are concatenated using pd.concat to create the merged dataframe. If the column names are not compatible, the CSV file is added to a list of incompatible files.

After processing all CSV files, the function checks if any files were not compatible. If there are incompatible files, it prints a message indicating the files that are not compatible. Otherwise, it prints a success message.

The merged dataframe is returned as the output of the function. In the provided code snippet, it is assigned to the variable df_merged for further usage



## Data Cleaning

### get column names

df.columns.to_list() retrieves the column names of a DataFrame df and converts them into a Python list.

The columns attribute of a DataFrame contains the column labels or names. By calling the to_list() method on the columns attribute, the column names are extracted and converted into a list.

# get column names
 a function rename_columns that renames the columns of a DataFrame based on a provided dictionary. It also includes additional steps to standardize the column names by replacing whitespace characters with underscores and converting them to lowercase.

The rename_columns function, easily renames the columns passing the DataFrame and a dictionary mapping the current column names to the desired new names. The function modifies the DataFrame in place and returns it.
