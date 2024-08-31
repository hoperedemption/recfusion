import pandas as pd 
import numpy as np

# read the dataset from the CSV file
book_df = pd.read_csv('../datasets/book_reviews/archive/Ratings.csv')

# find out the number of users using User-ID
first_user_id = book_df['User-ID'].min()
last_user_id = book_df['User-ID'].max()
num_users = last_user_id - first_user_id + 1

# filter out users without ratings
filtered_df = book_df[book_df['Book-Rating'] >= 1]

# replace 0 ratings with NaNs
book_df['Book-Rating'] = book_df['Book-Rating'].replace(0, np.nan)

