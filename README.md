# music-recommender-system
A music recommender system that will recommend new musical artists to a user based on their listening history. Suggesting different songs or musical artists to a user is important to many music streaming services, such as Pandora and Spotify. In addition, this type of recommender system could also be used as a means of suggesting TV shows or movies to a user (e.g., Netflix). To create this system we use Spark and the collaborative filtering technique.

# Datasets
We will be using some publicly available song data from audioscrobbler, which can be found here (http://wwwetud.iro.umontreal.ca/~bergstrj/audioscrobbler_data.html). However, we modified the original data files so that the code will run in a reasonable time on a single machine. The reduced data files have been suffixed with _small.txt and contains only the information relevant to the top 50 most prolific users (highest artist play counts).

The original data file user_artist_data.txt contained about 141,000 unique users, and 1.6 million unique artists. About 24.2 million users’ plays of artists are recorded, along with their count.

Note that when plays are scribbled, the client application submits the name of the artist being played. This name could be misspelled or nonstandard, and this may only be detected later. For example, "The Smiths", "Smiths, The", and "the smiths" may appear as distinct artist IDs in the data set, even though they clearly refer to the same artist. So, the data set includes artist_alias.txt, which maps artist IDs that are known misspellings or variants to the canonical ID of that artist. The artist_data.txt file then provides a map from the canonical artist ID to the name of the artist.

# The Recommender Model
For this project, we will train the model with implicit feedback. We can read more information about this from the collaborative filtering page: http://spark.apache.org/docs/latest/mllib-collaborative-filtering.html (http://spark.apache.org/docs/latest/mllibcollaborative-filtering.html). The function we will be using (http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.ALS.trainImplicit) has a few tunable parameters that will affect how the model is built. Therefore, to get the best model, we will do a small parameter sweep and choose the model that performs the best on the validation set 

Therefore, we must first devise a way to evaluate models. Once we have a method for evaluation, we can run a parameter sweep, evaluate each combination of parameters on the validation data, and choose the optimal set of parameters. The parameters then can be used to make predictions on the test data. 

# Model Evaluation
Although there may be several ways to evaluate a model, we will use a simple method here. Suppose we have a model and some dataset of true artist plays for a set of users. This model can be used to predict the top X artist recommendations for a user and these recommendations can be compared the artists that the user actually listened to (here, X will be the number of artists in the dataset of true artist plays). Then, the fraction of overlap between the top X predictions of the model and the X artists that the user actually listened to can be calculated. This process can be repeated for all users and an average value returned.

For example, suppose a model predicted [1,2,4,8] as the top X=4 artists for a user. Suppose, that user actually listened to the
artists [1,3,7,8]. Then, for this user, the model would have a score of 2/4=0.5. To get the overall score, this would be
performed for all users, with the average returned.

NOTE: when using the model to predict the top-X artists for a user, we do not include the artists listed with that user in
the training data.

Name the function modelEval and have it take a model (the output of ALS.trainImplicit) and a dataset as input. For parameter tuning, the dataset parameter should be set to the validation data (validationData). After parameter tuning, the model can be evaluated on the test data (testData).

# Model Construction
Now we can build the best model possibly using the validation set of data and the modelEval function. Although, there are a few parameters we could optimize, for the sake of time, we will just try a few different values for the rank parameter (http://spark.apache.org/docs/latest/mllib-collaborative-filtering.html#collaborative-filtering) (leave everything else at its default value, except make seed=345). Loop through the values [2, 10, 20] and figure out which one produces the highest scored based on the model evaluation function.

Now, using the bestModel, we will check the results over the test data.

# Trying Some Artist Recommendations
Using the best model above, predict the top 5 artists for user 1059637 using the recommendProducts (http://spark.apache.org/docs/1.5.2/api/python/pyspark.mllib.html#pyspark.mllib.recommendation.MatrixFactorizationModel.recommendProducts) function and map the results (integer IDs) into the real artist name using artistAlias.
