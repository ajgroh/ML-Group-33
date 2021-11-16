CS 4641 Predicting Playoff Appearances
### Introduction
For our group project, we are trying to accurately predict whether or not a NFL team will make it into the playoffs based on their performance through the midway point of the regular season. In the NFL, there are 32 teams. Of those 32 teams, 14 teams will make the playoffs. There are two conferences in the NFL, the NFC and the AFC which each contain 16 teams. The conferences are further split into 4 divisions each (8 divisions total). The top team (based on record) in each division automatically make the playoffs. The division champions account for 8 out of the 14 teams that make the playoffs each year. The other 6 teams are wildcards. The three teams with the best record in each conference (after the teams that were division champions are removed) fill the remaining spots in the playoffs (3 teams from each conference fill the last 6 spots). The NFL regular season is 18 weeks long where each team plays 17 games because of one bye week. We have reason to believe that by week 9 in the regular season, a machine learning model could predict with relative accuracy whether or not a team will make it into the playoffs that year. 

### Background
By creating a model that provides teams with a reasonable estimation on whether or not they will make it into the playoffs, we are providing actionable information. Teams need these predictions to make changes to their roster, budget, as well as decide whether or not to settle for a higher draft pick. For example, if a team uses this model to find out that they are most likely not making it into the playoffs, the administration can focus on trading players to better position the team for the following season. In regards to budgetting, this model could help teams prepare the stadium for Playoff season by allowing them to account for those preparations in their budget weeks in advance. Because of the structure setup by the NFL, teams do not get to keep the money from ticket sales during the playoff season. Therefore, teams can actually lose money during the playoffs; therefore, a model that can predict if a team will need to account for an increase in spending after the regular season would benefit NFL teams immensely. Also, if a team is not predicted to make the playoffs by week 9, then the team could spend less money picking up players to save the current season. Instead, those teams can save money and finish with more losses. This will enable those teams to pick players ealier in the draft; therefore, they can get the best talent coming into the league. 

### Data Collection

 



For our group project proposal we are trying to accurately predict the record for all 32 NFL teams/final standings before playoffs based on preseason results. We will also take into account other important aspects such as trades between seasons, prior home and away records, as well as head to heads. We felt making these types of predictions is important because predicting performance is very difficult and it’s important to find the best data that will give the most accurate output. Teams need these predictions to make changes to their roster, budget, as well as decide whether or not to settle for a higher draft pick. These predictions not only impact the franchises themselves but also companies that specialize in sports betting and fantasy. We will take into account overall record, rushing yards, passing yards, interceptions, sacks, and more. Based on the paper “Predicting Margin of Victory in NFL Games: Machine Learning vs. the Las Vegas Line”, we will also include relevant features such as points per game, points per game allowed, yards per game and yards per game allowed from the footballdb dataset. Because starting players during the regular season typically play for every game in the preseason, we believe that data from the preseason will provide helpful insights into regular season performance. Our approach is somewhat different as most power rankings are solely based on the previous season record. We believe that taking both previous season records and preseason records into account will give us more accurate results. One of our sources that we will be using is The Football Database which gives us stats from any season regular and preseason as well as detailed stats on every player and the overall game. We will evaluate our approach and results based on how closely our team record predictions are to the actual game records for the season. The timeline works well in that the regular season ends in January, therefore we will be able to see how our model is performing in real time. A few things that can definitely have an impact on our results are injuries and player trades during the season. It’s very hard to predict how a team will respond based on a key player injury. It might also be hard to adapt and change our method if, for example, a quarterback gets injured and there isn’t much data on the team’s second string quarterback. These unexpected events during the season could cause errors in the model. One possible solution to a problem like that could be looking at an injury from a previous season and seeing how teams were able to adjust by looking at their record and stats following that injury. In conclusion, we will use a recurrent neural network to predict regular season performance by training our model on preseason data. We chose an RNN as our model because this algorithm typically performs best on time series data (like preseason statistics over the past decade) and hopefully predict accurate season standings.





## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/rorbrow/MLtest/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/rorbrow/MLtest/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
