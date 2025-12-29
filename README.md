🏏 IPL Match Prediction using Machine Learning

Project Overview
This initiative centers on forecasting the eventual **winner of an IPL (Indian Premier League) match** by harnessing archival match records and statistical learning methods. The system ingests contextual match attributes—participating franchises, toss outcomes, playing venue, and seasonality—to infer the most plausible victor.

The work has been undertaken strictly for **academic exploration and conceptual reinforcement**, serving as a bridge between theory and applied machine learning.

Dataset Information
The backbone of this study is a comprehensive IPL Matches Data collection spanning **2008 through 2023**, encapsulated within the file `matches.csv`. This dataset chronicles seasons of competitive cricket, offering a fertile ground for pattern discovery.

Key Fields Utilized
The analytical focus is narrowed to the following variables:

* `team1`
* `team2`
* `toss_decision`
* `venue`
* `season`
* `winner` (designated as the response variable)

Features Considered for Prediction
To sculpt a meaningful prediction surface, the model evaluates:

* Competing Team 1
* Competing Team 2
* Decision made at the toss
* Stadium or ground where the match unfolds
* Season year, reflecting temporal dynamics

Machine Learning Model
At the core lies a **Logistic Regression** classifier, chosen for its interpretability and robustness with categorical-heavy datasets. Prior to training, categorical descriptors are transformed through **OneHotEncoding**, orchestrated via a **ColumnTransformer** integrated into a streamlined **Pipeline**.

The dataset is partitioned using an **80:20 train–test split**, ensuring a balanced assessment of generalization capability. Model efficacy is gauged using the **Accuracy Score**, providing a clear, quantitative performance signal.

Model Accuracy
Upon evaluation against the unseen test subset, the model records an accuracy of approximately **(add your accuracy here)%**, reflecting its competence in discerning match outcomes from historical signals.

How to Run the Project

Clone the Repository
To begin local execution, retrieve the project source using the following command:

```bash
git clone https://github.com/divyansh-dev8/IPL-MATCH-Prediction-.git
```
