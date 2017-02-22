# churn_case_study
Case Study - Churn Prediction
A ride-sharing company (Company X) is interested in predicting rider retention. To help explore this question, we have provided a sample dataset of a cohort of users who signed up for an account in January 2014. The data was pulled on July 1, 2014; we consider a user retained if they were “active” (i.e. took a trip) in the preceding 30 days (from the day the data was pulled). Assume the latest day of last_trip_date to be when the data was pulled. The data is churn.csv in the data folder.

We would like you to use this data set to help understand what factors are the best predictors for retention, and offer suggestions to operationalize those insights to help Company X. Therefore, your task is not only to build a model that minimizes error, but also a model that allows you to interpret the factors that contributed to your predictions.

Here is a detailed description of the data:

city: city this user signed up in
phone: primary device for this user
signup_date: date of account registration; in the form `YYYYMMDD`
last_trip_date: the last time this user completed a trip; in the form `YYYYMMDD`
avg_dist: the average distance (in miles) per trip taken in the first 30 days after signup
avg_rating_by_driver: the rider’s average rating over all of their trips
avg_rating_of_driver: the rider’s average rating of their drivers over all of their trips
surge_pct: the percent of trips taken with surge multiplier > 1
avg_surge: The average surge multiplier over all of this user’s trips
trips_in_first_30_days: the number of trips this user took in the first 30 days after signing up
luxury_car_user: TRUE if the user took a luxury car in their first 30 days; FALSE otherwise
weekday_pct: the percent of the user’s trips occurring during a weekday
Work Flow
Perform any cleaning, exploratory analysis, and/or visualizations to use the provided data for this analysis.

Build a predictive model to help determine whether or not a user will be retained.

Evaluate the model

Identify / interpret features that are the most influential in affecting your predictions

Discuss the validity of your model. Issues such as leakage

Repeat 2 - 5 until you have a satisfactory model

Deliverables
Code you used to build the model (submit via pull request)

A (verbal) presentation including the following points:

How did you compute the target?
What model did you use in the end? Why?
Alternative models you considered? Why are they not good enough?
What performance metric did you use? Why?
Based on insights from the model, what actionable plans do you propose to reduce churn?
