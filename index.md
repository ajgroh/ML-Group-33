# CS 4641 Group 33: Predicting Playoff Appearances

### Introduction/Background
For our group project, we are trying to accurately predict whether or not a NFL team will make it into the playoffs based on their performance through the midway point of the regular season. In the NFL, there are 32 teams. Of those 32 teams, 14 teams will make the playoffs. There are two conferences in the NFL, the NFC and the AFC which each contain 16 teams. The conferences are further split into 4 divisions each (8 divisions total). The top team (based on record) in each division automatically make the playoffs. The division champions account for 8 out of the 14 teams that make the playoffs each year. The other 6 teams are wildcards. The three teams with the best record in each conference (after the teams that were division champions are removed) fill the remaining spots in the playoffs (3 teams from each conference fill the last 6 spots) [3]. The NFL regular season is 18 weeks long where each team plays 17 games because of one bye week. Another important aspect of the NFL is the NFL Draft. A teams order in the NFL Draft is determined by their previous season performance. The team who does the worst gets the first pick in each round, and the Super Bowl winner gets the last pick in each round [2]. 

<p align="center">
  <img src="https://images.squarespace-cdn.com/content/v1/5ce3ea91ae2b190001d02fba/1601444797075-M9N6ZBUUCKQZV1FD1CHC/Teams+by+division.jpg" alt="NFL Teams By Division"/>
</p>

### Problem Statement
By creating a model that provides teams with a reasonable estimation on whether or not they will make it into the playoffs, we are providing actionable information. Teams need these predictions to make changes to their roster, budget, as well as decide whether or not to settle for a higher draft pick. For example, if a team uses this model to find out that they are most likely not making it into the playoffs, the administration can focus on trading players to better position the team for the following season. In regards to budgetting, this model could help teams prepare the stadium for Playoff season by allowing them to account for those preparations in their budget weeks in advance. Because of the structure setup by the NFL, teams do not get to keep the money from ticket sales during the playoff season. This means teams can actually lose money during the playoffs; therefore, a model that can predict if a team will need to account for an increase in spending after the regular season would benefit NFL teams immensely. Also, if a team is not predicted to make the playoffs by week 9, the team could spend less money picking up players in an attempt to save the current season. There is another benefit to knowing if your team will not make the playoffs: next seasons draft seeding. Intentionally throwing a season after using our model to learn the team won't make the playoffs will cause that team to pick players ealier in the draft, enabling them to get the best talent coming into the league. We believe that by week 9 in the regular season, a machine learning model could predict with relative accuracy whether or not a team will make it into the playoffs that year. 

### Data Collection

For this project we manually retreived our data from multiple datasets on the Pro Football Reference Website [1]. Since the layout of the website requires the user to manually download each teams data for every season, we downloaded a csv file for each datapoint. Then, we used a python script to read and format the data into a final csv (NFLData.csv). This data contains one column for each feature and one column for our labels. The features we used were derived from the csv data files that we downloaded. We average the data from each team's performance in the first 9 weeks of every season. These averages are saved in the data file along with a 1 or 0 in the label column. The 1 represents that team making the playoffs that season, and a 0 represents that team failing to make the playoffs that season. All features used from this dataset are floating point numbers. When reading from the data csv's coming from Pro Football Reference, we dropped data features like week (of the season), day of the week, and time of day as we do not think that these features would impact a team's playoff possibility. We also dropped the feature focused on whether or not that game went into overtime. A problem we ran into while processing data from these sources was the possibility a bye week falling within the first 9 weeks of the regular season. This was represented by a row of Nan entries. In order to clean these entries from our dataset, we loop through each csv file and remove a row if there is a Nan present. In order to better understand our data, we first reduced the dimensionality of our data to 3D and graphed to view any major discrepencies or unexepected patterns/outliers. The graph of our model is shown below:
<p align="center">
  <img src="/img/pca_data_graph.png" alt="PCA Visualization" style="width: 30%">
</p>
An interesting characteristic of our dataset is the number of datapoints labeled with 1's vs the number of datapoints labeled with 0's. Because our dataset covers the past three years of every NFL team's season, there are more teams that do not make the playoffs (more 0's) then there are teams that make the playoffs (1's). This discrepency can affect the error in our model because our models could learn to classify every point as a 0 and still maintain an acceptable accuracy (above complete random 50/50). To address this problem, we believe that we need to get more data so that our model will be able to accurately classify points and not get trapped predicting all 0's. Our next goal is to add to our dataset by creating a webscraping script that automatically pulls the necessary data off the Pro Football Reference website and formats that labeled data into a csv. This would save a lot of manual labeling and pulling of data and allow us to greatly expand our dataset. If the problem persists even after the dataset is grown, we will most likely remove some datapoints that are labeled as 0's to ensure that approximately 50% are labeled 0 and 50% are labeled 1. 


### Methods

### Results

Both Logistic Regression with PCA and Logistic Regression without PCA have variable accuracies between 0.667 and 0.91667 for our model, whereas Linear Regression without PCA has a bit lower worse-case of 0.625, and the same best-case of 0.91667. The reason the accuracies vary is because we do not have static training and testing data sets. We randomize which part of our dataset is testing and which part is training each time we run the algorithm, which helps us see how robust our model is and if we are overfitting. 

<p align="center">
  Best-Case Logistic Regression with PCA: </br>
  
  <img src="/img/PCAGood.png" alt="GoodPCA">
</p>

<p align="center">
  Worst-Case Logistic Regression with PCA: </br>
  
  <img src="/img/PCABad.png" alt="BadPCA">
</p>

<p align="center">
  Best-Case Logistic Regression without PCA: </br>
  
  <img src="/img/LogRegGood.png" alt="GoodLogReg">
</p>

<p align="center">
  Worst-Case Logistic Regression without PCA: </br>
  
  <img src="/img/LogRegBad.png" alt="BadLogReg">
</p>

<p align="center">
  Best-Case Linear Regression without PCA: </br>
  
  <img src="/img/LinRegGood.png" alt="GoodLinReg">
</p>

<p align="center">
  Worst-Case Linear Regression without PCA: </br>
  
  <img src="/img/LinRegBad.png" alt="BadLinReg">
</p>

Right now it seems like we are experiencing overfitting a bit due to the variance in accuracy, and this is likely due to our limited dataset. As previously stated in the Data Collection section, we will write a script in order to get more data from a less user-friendly website. We think this will have a huge impact on our model.

### Discussion

So far, a couple of our models are producing quite impressive accuracies, but the inconsistency lies in the randomization of test vs training data per iteration. As mentioned above, we are planning on writing a script in order to download even more data to cleanse and use for training and testing. We think that this will make the accuracies produced by the randomized test/training sets much more consistent, which will show us that our model is robust. This new data may also change how accurate our current models are, and will likely cause us to reconsider our chosen algorithms and/or tweak hyperparameters. If we are still able to achieve high accuracy after adding all of this new data, we think it may be a good idea to try to reduce the number of weeks we use to predict the playoff appearances. If we are able to reduce the number of weeks, that would make our model even more useful, as the beneficiaries of this model would be able to make predictions even earlier in the season and a lot of the decisions made from this data are time sensitive.


### References
  [1] https://www.pro-football-reference.com/ <br/>
  [2] https://www.sportingnews.com/us/nfl/news/how-does-the-nfl-draft-work-rules-rounds-eligibility-and-more/o431yshp0l431e7543pcrzpg1 <br/>
  [3] https://www.si.com/nfl/2021/01/09/nfl-expanded-playoffs-explained <br/>

