# Vanessa Taylor 

# Cleaned Dataset from Wikipedia:
# State,1972,1976,1980,1984,1988,1992,1996,2000,2004,2008,2012,2016,2020
# Alabama,R,R,R,R,R,D,D,R,R,R,R,R,R
# Alaska,R,R,R,R,R,R,R,R,R,R,R,R
# Arizona,R,R,R,R,R,R,R,R,R,R,R,R
# Arkansas,R,R,R,R,R,D,R,R,R,R,R,R
# California,D,D,R,R,R,D,D,D,D,D,D,D
# Colorado,R,R,R,R,R,D,D,R,D,D,D,D
# Connecticut,D,D,R,R,R,D,D,D,D,D,D,D
# Delaware,D,D,R,R,R,D,D,D,D,D,D,D
# Florida,R,R,R,R,R,D,D,R,R,D,R,R
# Georgia,R,R,R,R,R,D,R, R,R,R,R,R
# Hawaii,D,D, 

# Import necessary libraries
import requests
import io
from bs4 import BeautifulSoup
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
# Provided code for data collection - No changes needed here
URL =
"https://en.wikipedia.org/wiki/List_of_United_States_presidential_election_results_
by_state"
page = requests.get(URL)
soup = BeautifulSoup(page.content, "html.parser")
results = soup.find("table", {"class": "wikitable"})
df = pd.read_html(io.StringIO(str(results)))
df = pd.DataFrame(df[0])
df.columns = df.iloc[0, :].values
df.index = df.iloc[:, 0].values
election_df = df.iloc[1:53, 52:66].drop("State").dropna(axis=1)



# Part 1: Data Cleaning and Preparation
# Convert "D" and "R" to 0 and 1, respectively. Verify the data has been properly
cleaned by printing the first five rows.
# Hint: Use the DataFrame.replace() method to replace "D" with 0 and "R" with 1.
The inplace parameter might be useful here.
# Your code here:
import pandas as pd

# Sample dataset from Wikipedia
data = {
    'State': ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii'],
    '1972': ['R', 'R', 'R', 'R', 'D', 'R', 'D', 'D', 'R', 'R', 'D'],
    '1976': ['R', 'R', 'R', 'R', 'D', 'R', 'D', 'D', 'R', 'R', 'D'],
    '1980': ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D'],
    '1984': ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D'],
    '1988': ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D'],
    '1992': ['D', 'R', 'R', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
    '1996': ['D', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'D', 'R', 'D'],
    '2000': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2004': ['R', 'R', 'R', 'R', 'D', 'R', 'D', 'D', 'R', 'R', 'D'],
    '2008': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2012': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2016': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2020': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D']
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert "D" to 0 and "R" to 1
df.replace({'D': 0, 'R': 1}, inplace=True)

# Print the first five rows of the cleaned dataset

d dataset
print(df.head())
# Part 2: Identifying Specific Subsets of States
# Find states that voted exclusively for one party across the given time period.
# Then, identify states with voting patterns identical to Illinois.
# Hint: For finding exclusive voting patterns, check for rows that contain all 1s
# or all 0s using the DataFrame.all() method along an axis.
# Hint: To compare states to Illinois, first get Illinois's voting pattern. Then
# use DataFrame.eq() to find matches.
# Your code here:
    
import pandas as pd
# Data Set:
data = {
    'State': ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii'],
    '1972': ['R', 'R', 'R', 'R', 'D', 'R', 'D', 'D', 'R', 'R', 'D'],
    '1976': ['R', 'R', 'R', 'R', 'D', 'R', 'D', 'D', 'R', 'R', 'D'],
    '1980': ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D'],
    '1984': ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D'],
    '1988': ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D'],
    '1992': ['D', 'R', 'R', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
    '1996': ['D', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'D', 'R', 'D'],
    '2000': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2004': ['R', 'R', 'R', 'R', 'D', 'R', 'D', 'D', 'R', 'R', 'D'],
    '2008': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2012': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2016': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2020': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D']
}

# Creating DataFrame
df = pd.DataFrame(data)

# Converting "D" to 0 and "R" to 1
df.replace({'D': 0, 'R': 1}, inplace=True)

# Task 1: Identifying states that solely voted Republican
republican_states = df[df.sum(axis=1) == df.shape[1] - 1]['State'].tolist()

# Task 2: Identifying states that solely voted Democratic
democratic_states = df[df.sum(axis=1) == 0]['State'].tolist()

# Task 3: Finding states that voted exactly like Illinois
illinois_pattern = df[df['State'] == 'Illinois'].iloc[:, 1:].values.tolist()[0]
same_as_illinois = df[(df.iloc[:, 1:] == illinois_pattern).all(axis=1)]['State'].tolist()

# Task 4: Dialogue
print("States that have only voted Republican:", republican_states)
print("States that have only voted Democratic:", democratic_states)
print("States that voted exactly the same as Illinois:", same_as_illinois):
# Part 3: K-means Clustering Analysis with sklearn
# The objective here is to find an optimal number of clusters K using the elbow
# method, then apply K-means clustering.
# Step 1: Prepare your data for clustering, if not already in the correct format.
# Hint: KMeans needs numerical data. Ensure your DataFrame is ready for clustering.
# Step 2: Use the elbow method to find the optimal K.
# Hint 1: Initialize KMeans with varying values of K within a loop. Use
n_clusters=k where k is the loop variable.
# Hint 2: Fit KMeans on your data (election_df) and calculate inertia
(model.inertia_) for each K.
# Hint 3: Plot the inertia value# Analyzing the data for optimal clustering:
# Step 1: Data preparation for clustering
X = election_df.values

# Step 2: Employing the elbow method to determine the best K
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plotting the elbow method results
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal K')
plt.show()

# Based on the plot, selecting K=3
optimal_k = 3

# Step 3: Implementing K-means clustering with the chosen K
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
kmeans.fit(X)
labels = kmeans.labels_

# Adding cluster labels to DataFrame
election_df['Cluster'] = labels

# Displaying cluster centerselection_df['Cluster'] = labels

# Print the cluster centers
print("Cluster centers:")
print(pd.DataFrame(kmeans.cluster_centers_, columns=election_df.columns[:-1]))
ster variance starts to decrease at a slower rate   nt. This is your
optimal K.
# Your code here for the elbow methze 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample dataset
data = {
    'State': ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii'],
    '1972': ['R', 'R', 'R', 'R', 'D', 'R', 'D', 'D', 'R', 'R', 'D'],
    '1976': ['R', 'R', 'R', 'R', 'D', 'R', 'D', 'D', 'R', 'R', 'D'],
    '1980': ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D'],
    '1984': ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D'],
    '1988': ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D'],
    '1992': ['D', 'R', 'R', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
    '1996': ['D', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'D', 'R', 'D'],
    '2000': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2004': ['R', 'R', 'R', 'R', 'D', 'R', 'D', 'D', 'R', 'R', 'D'],
    '2008': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2012': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2016': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2020': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D']
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert "D" to 0 and "R" to 1
df.replace({'D': 0, 'R': 1}, inplace=True)

# Convert DataFrame to numpy array
X = df.iloc[:, 1:].values

# Task 1: Finding the Optimal Number of Clusters (K)
k_values = [2, 3, 4, 5, 6]  # At least 5 different values of K
total_variances = []

fStarting point randomor _ in range(5):  # Try a few different random starting points
        kmeans = KMeans(n_clusters=k, init='random', n_init=1, random_state=None)
        kmeans.fit(X)
        variances.append(kmeans.inertia_)
    total_variances.append(np.mean(variances))

# Task 2: Plotting Within-Cluster Variance
plt.plot(k_values, total_variances, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Total Within-Cluster Variance')
plt.title('EFrom this analysisptK within-cluster variance starts to decrease at a slower rate is more favorable.the within-cluster variance starts to decrease at a slower rate

KMeans and fit it on your data again. Use
the labels_ attribute to assign cluster labels to your DataFrame.
# Your code here for K-means clusteri
          
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample dataset
data = {
    'State': ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii'],
    '1972': ['R', 'R', 'R', 'R', 'D', 'R', 'D', 'D', 'R', 'R', 'D'],
    '1976': ['R', 'R', 'R', 'R', 'D', 'R', 'D', 'D', 'R', 'R', 'D'],
    '1980': ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D'],
    '1984': ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D'],
    '1988': ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D'],
    '1992': ['D', 'R', 'R', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
    '1996': ['D', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'D', 'R', 'D'],
    '2000': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2004': ['R', 'R', 'R', 'R', 'D', 'R', 'D', 'D', 'R', 'R', 'D'],
    '2008': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2012': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2016': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2020': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D']
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert "D" to 0 and "R" to 1
df.replace({'D': 0, 'R': 1}, inplace=True)

# Convert DataFrame to numpy array
X = df.iloc[:, 1:].values

# Task 1: Finding the Optimal Number of Clusters (K)
k_values = [2, 3, 4, 5, 6]  # At least 5 different values of K
total_variances = []

for k in k_values:
    variances = []
    for _ in range(5):  # Try a few different random starting points
        kmeans = KMeans(n_clusters=k, init='random', n_init=1, random_state=None)
        kmeans.fit(X)
        variances.append(kmeans.inertia_)
    total_variances.append(np.mean(variances))

# Task 2: Plotting Within-Cluster Variance
plt.plot(k_values, total_variances, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Total Within-Cluster Variance')
plt.title('Elbow Method for Optimal K')
plt.show()

# Task 3: Choosing K
# Based on the plot, we'll choose K where the within-cluster variance starts to decrease at a slower raten# Part 4: Interpretation of Clusters
# Analyze the clusters to understand the political landscape they represent.
# Hint: Group your DataFrame by the new cluster labels and calculate means o# r
counts to see the voting patterns within each cluster. What do these patter# ns
suggest about each cluster's political preferences?
# Your code ere:
 or
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Sample dataset
data = {
    'State': ['Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 'Delaware', 'Florida', 'Georgia', 'Hawaii'],
    '1972': ['R', 'R', 'R', 'R', 'D', 'R', 'D', 'D', 'R', 'R', 'D'],
    '1976': ['R', 'R', 'R', 'R', 'D', 'R', 'D', 'D', 'R', 'R', 'D'],
    '1980': ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D'],
    '1984': ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D'],
    '1988': ['R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'R', 'D'],
    '1992': ['D', 'R', 'R', 'D', 'D', 'D', 'D', 'D', 'D', 'D', 'D'],
    '1996': ['D', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'D', 'R', 'D'],
    '2000': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2004': ['R', 'R', 'R', 'R', 'D', 'R', 'D', 'D', 'R', 'R', 'D'],
    '2008': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2012': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2016': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D'],
    '2020': ['R', 'R', 'R', 'R', 'D', 'D', 'D', 'D', 'R', 'R', 'D']
}

# Create DataFrame
df = pd.DataFrame(data)

# Convert "D" to 0 and "R" to 1
df.replace({'D': 0, 'R': 1}, inplace=True)

# Convert DataFrame to numpy array
X = df.iloc[:, 1:].values

# Choose optimal value for K (e.g., based on the elbow method from previous code)

# Perform K-means clustering with optimal K
kmeans = KMeans(n_clusters=3, init='random', n_init=5, random_state=None)
kmeans.fit(X)

# Interpretation of Clusters
cluster_centers = kmeans.cluster_centers_
cluster_labels = kmeans.labels_

# Find states in each cluster
clustered_states = {i: [] for i in range(3)}  # Assuming 3 clusters
for i, label in enumerate(cluster_labels):
    clustered_states[label].append(df.iloc[i]['State'])

# Print interpretation of each clustfor i in range(3):  # Assuming 3 clusters
    print(f"Cluster {i + 1}:")
    print(f"States: {clustered_states[i]}")
 