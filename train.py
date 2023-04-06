# Import required libraries
import pandas as pd
from surprise import Dataset, Reader, KNNBasic, KNNWithMeans
from surprise.model_selection import train_test_split, cross_validate
from surprise.accuracy import rmse, mae

# Load the data into a Pandas dataframe
data = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Create a reader object for Surprise
reader = Reader(rating_scale=(1, 5))

# Load the data into a Surprise dataset
dataset = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)

# Split the data into training and testing sets
trainset, testset = train_test_split(dataset, test_size=0.25)

# Train and evaluate a user-based collaborative filtering model
user_based = KNNBasic(sim_options={'user_based': True})
user_based.fit(trainset)
user_based_predictions = user_based.test(testset)
user_based_rmse = rmse(user_based_predictions)
user_based_mae = mae(user_based_predictions)
print('User-based CF RMSE: {:.4f}'.format(user_based_rmse))
print('User-based CF MAE: {:.4f}'.format(user_based_mae))

# Train and evaluate an item-based collaborative filtering model
item_based = KNNBasic(sim_options={'user_based': False})
item_based.fit(trainset)
item_based_predictions = item_based.test(testset)
item_based_rmse = rmse(item_based_predictions)
item_based_mae = mae(item_based_predictions)
print('Item-based CF RMSE: {:.4f}'.format(item_based_rmse))
print('Item-based CF MAE: {:.4f}'.format(item_based_mae))

# Train and evaluate a user-based collaborative filtering model with mean centering
user_based_means = KNNWithMeans(sim_options={'user_based': True})
user_based_means.fit(trainset)
user_based_means_predictions = user_based_means.test(testset)
user_based_means_rmse = rmse(user_based_means_predictions)
user_based_means_mae = mae(user_based_means_predictions)
print('User-based CF with Means RMSE: {:.4f}'.format(user_based_means_rmse))
print('User-based CF with Means MAE: {:.4f}'.format(user_based_means_mae))
