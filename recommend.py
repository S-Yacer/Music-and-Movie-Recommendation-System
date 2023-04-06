# Import required libraries
import pandas as pd
from surprise import Dataset, Reader, KNNWithMeans

# Load the data into a Pandas dataframe
data = pd.read_csv('ml-100k/u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])

# Create a reader object for Surprise
reader = Reader(rating_scale=(1, 5))

# Load the data into a Surprise dataset
dataset = Dataset.load_from_df(data[['user_id', 'item_id', 'rating']], reader)

# Train a user-based collaborative filtering model with mean centering
user_based_means = KNNWithMeans(sim_options={'user_based': True})
trainset = dataset.build_full_trainset()
user_based_means.fit(trainset)

# Define a function to generate top-N recommendations for a given user
def get_top_n(user_id, n=10):
    # Get the list of all item IDs
    all_item_ids = list(trainset.all_items())

    # Remove the items that the user has already rated
    rated_item_ids = [r.iid for r in trainset.ur[user_id]]
    unrated_item_ids = list(set(all_item_ids) - set(rated_item_ids))

    # Predict the ratings for the unrated items
    testset = [(user_id, item_id, 4) for item_id in unrated_item_ids]
    predictions = user_based_means.test(testset)

    # Sort the predictions by predicted rating in descending order
    top_n = sorted(predictions, key=lambda x: x.est, reverse=True)[:n]

    # Return the top-N recommended item IDs and predicted ratings
    top_n_item_ids = [r.iid for r in top_n]
    top_n_ratings = [r.est for r in top_n]
    return top_n_item_ids, top_n_ratings

# Example usage: generate top-10 recommendations for user 5
top_n_item_ids, top_n_ratings = get_top_n(5, n=10)
print('Top-10 recommended item IDs for user 5: {}'.format(top_n_item_ids))
print('Predicted ratings for the top-10 recommended items: {}'.format(top_n_ratings))
