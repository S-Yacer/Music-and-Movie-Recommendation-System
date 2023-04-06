
# Music and Movie Recommendation System

This is a collaborative filtering recommendation system for music and movie ratings, implemented using Python and the Surprise library. The system is trained on the public MovieLens 100k dataset and can be used to generate top-N recommendations for a given user.

# Getting Started

To use this recommendation system, you will need to have Python 3 and the following Python packages installed:
```
pandas
surprise
```
You can install these packages using pip by running the following command:

```
pip install pandas surprise
```
Once you have installed the required packages, you can train the recommendation system by running the train.py script:

```
python train.py
```
This script will load the MovieLens 100k dataset, train a user-based collaborative filtering model with mean centering, and save the trained model to a file called model.pkl.

You can then generate top-N recommendations for a given user by running the recommend.py script:

```
python recommend.py <user_id> <n>
```
Replace ```<user_id>``` with the ID of the user for whom you want to generate recommendations, and replace ```<n>``` with the number of recommendations you want to generate.

# Acknowledgments
This project is based on the MovieLens 100k dataset, which was created by GroupLens Research at the University of Minnesota. The dataset can be found at http://grouplens.org/datasets/movielens/100k/.

# Future Work
Implement item-based collaborative filtering.
Explore the use of additional features such as movie genres and user demographics.
Implement a web interface for the recommendation system.
