import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    # Read-in csv files
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # merge Dataframes by 'id'
    df = messages.merge(categories, on='id')
    return df


def clean_data(df):
    # creating a dataframe of individual category columns
    categories = df.categories.str.split(';',expand=True)
    
    # selecting the first row of the categories dataframe to extract new column names for categories
    row = categories.iloc[0]
    category_colnames = [x[:-2] for x in row]
    
    # renaming the columns of `categories`
    categories.columns = category_colnames
    
    # setting each value of categories to be the last character of the string
    for column in categories:
        categories[column] = categories[column].map(lambda x: str(x)[-1:])
        # converting column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    #Replacing categories column in df with new category columns    
    df.drop(['categories'],axis=1,inplace=True)
    df=pd.concat([df,categories],axis=1)
    
    #Removing duplicates
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    database_path = 'sqlite:///' + database_filename
    engine = create_engine(database_path)
    database_name = database_filename.split('/')[-1].split('.')[0] # split given path to get just the name
    df.to_sql(database_name, engine, index=False) 
    print(database_name)
    return  


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()