# Data

This folder contains scripts and notebooks used to modify the data set that was provided by the Polish Cyber command into 
the datasets we have used in our application. A summary of the processes can be found below

## Dataset modification

- Concatinated all twitter datasets
- Removed duplicates
- Added a column of hashtag free text
- Extracted user information on a more readable format and added a column to dataframe for each data point
- Calculated the age of the account at the time of each tweet

Pseudo labelling of the full data can be done based on the hashtags and semantic likeness to known propaganda talking points.

For a small subset, we manually verified 50 instances of misinformation in the dataset and made a demo dataset containing those 50 samples 
as well as 100 random samples

## Labeling Policy

TODO: How did we pseudo label: which hashtags, semantic likeness to verified false information etc 