The challenge was posted on [HackerEarth](https://www.hackerearth.com/challenge/competitive/predict-the-happiness/)  as a Machine Learning challenge for beginners series. The aim of the challenge was to predict the sentiments of various reviews.

I secured third position on the [leaderboard](https://www.hackerearth.com/challenge/competitive/predict-the-happiness/leaderboard/). This part of the repository contains a notebook showing  various models I experimented with and my final approach for the challenge. 

To summarize, I used a combination of a deep ConvNet with an Embedding layer and XGBoost. The ConvNet and Embedding layer combination was first trained on text revies in the [TripAdvisor dataset](http://times.cs.uiuc.edu/~wang296/Data/). The dataset provided for the challenge was used to finetune this model. Finally I used an XGBoost classifer which took in the probability scores given by the ConvNet and other features in the dataset, namely, 'Browser Used' and 'Device Used' and used it's predictions as my submission. 

A detailed blog post for the same is coming soon!
