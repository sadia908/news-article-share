import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB

# Read the data in and take a look at it
data = pd.read_csv("OnlineNewsPopularity.csv")

# The columns appear to have a space at the beginning of the name, let's remove that
data.columns = data.columns.str.replace(" ", "")

# Remove the two descriptive variables which are not used for prediction
data.drop(["url", "timedelta"], axis=1, inplace=True)

# Heatmap showing the correlation between variables
data_correlation = data.corr()
mask = np.triu(np.ones_like(data_correlation, dtype=bool))
sns.heatmap(data_correlation, mask=mask, cmap="Blues")


min_shares = data["shares"].min() - 1
shares_50 = data["shares"].quantile(0.50)
max_shares = data["shares"].max() + 1

popularity = pd.cut(
    data["shares"], bins=[min_shares, shares_50, max_shares], labels=["Low", "High"]
)

data["Popularity"] = popularity

# ### Naive Bayes, No Processing

y = data["Popularity"]
X = data.iloc[:, 0:-4]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=273
)

BNB = BernoulliNB()

BNB.fit(X_train, y_train)

# y_pred = BNB.predict([data])
y_pred = BNB.predict(X_test)
print(accuracy_score(y_test, y_pred))
