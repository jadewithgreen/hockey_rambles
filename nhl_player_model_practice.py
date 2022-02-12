""" python3
    Going through some models and analyzing the results for explorative purposes.
"""
# import all the packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy
import sklearn
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AffinityPropagation
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MaxAbsScaler


#read in cleaned player data into dataframes
f = pd.read_csv("rel_forward_offensive_data_for_pca.csv")
d = pd.read_csv("rel_defenseman_offensive_data_for_pca.csv")

""" According to the website nhlsguides.net/player-builds, there
are 6 different types of forwards:

Enforcer, Grinder, Playmaker, Power, Sniper, Two-Way

and 4 kinds of defensemen:

Defensive, Enforcer, Offensive, Two-Way

However, since I began this wanting to know possible offensive classes, and those
categories cover (presumably) the combination of two different classes
(defensive capabilities + offensive capabilities), I will have to try a variety of
different number of classes, from 3 to 6, with the possibility of expanding
out that range. My assumptions leading to this are:
- There are at least 3 types of offensive capabilities (good, mediocre, bad) at
  even strength
- At least some of the known player types (which are typically described in a subjective manner)
  can be derived from numerical data, which is a mix of "offensive" data and
  "defensive data"

To estimate the number of clusters, I'm going to try using kmeans and look at the
plots.
"""

X_test_05 = f[f["Season"] == "2005-06"]
X_test_05 = X_test_05.drop(columns = ["Season"])

"""
I now need to scale/preprocess the data before putting it into the model. Because
of the great differences in feature variation, I was unsure which scaler to use,
so I tried several.
"""
#normalization approach to scaling
norm = MinMaxScaler().fit(X_test_05)
X_test_norm_05 = norm.transform(X_test_05)
#max abs scaler
max_abs = MaxAbsScaler().fit(X_test_05)
X_test_max_abs_05 = max_abs.transform(X_test_05)
#standardization approach to scaling
standscale = StandardScaler().fit(X_test_05)
X_test_stand_05 = standscale.transform(X_test_05)
#quantile transformer scaler
#quant = QuantileTransformer().fit(X_test_05)
#X_test_quant_05 = quant.transform(X_test_05)
#robust scaler
robust = RobustScaler().fit(X_test_05)
X_test_rob_05 = robust.transform(X_test_05)
#power transformer scaler
powtrans = PowerTransformer().fit(X_test_05)
X_test_pow_05 = powtrans.transform(X_test_05)

#run PCA
pca = PCA(n_components = 4)
#print to check the different scaling methods

X_reduced = pca.fit(X_test_norm_05).transform(X_test_norm_05)
print("explained variance ratios with normalizer: %s"% str(pca.explained_variance_ratio_))
# 80% with first 2 features
X_reduced1 = pca.fit(X_test_stand_05).transform(X_test_stand_05)
print("explained variance ratios with standardizer: %s"% str(pca.explained_variance_ratio_))
# 79% with first 2 features
X_reduced2 = pca.fit(X_test_max_abs_05).transform(X_test_max_abs_05)
print("explained variance ratios with max_abs scaler: %s"% str(pca.explained_variance_ratio_))
# 86% with first 2 features
#X_reduced3 = pca.fit(X_test_quant_05).transform(X_test_quant_05)
#print("explained variance ratios with quantile transformer scaler: %s"% str(pca.explained_variance_ratio_))
# 79% with first 2
X_reduced4 = pca.fit(X_test_rob_05).transform(X_test_rob_05)
print("explained variance ratios with robust scaler: %s"% str(pca.explained_variance_ratio_))
#79% with first 2
x_reduced5 = pca.fit(X_test_pow_05).transform(X_test_pow_05)
print("explained variance ratios with power transformer scaler: %s"% str(pca.explained_variance_ratio_))
# 79% with first 2

"""
Perhaps surprisingly, the max_abs scaler did the best at flattening the four original features
down to 2 at 86% (of 100% for all features).

We'll move ahead with the MaxAbsScaler here then. Theory about why this scaler might be best
for this set:
It does not destroy sparsity since it does not shift/center the data. (I initially thought
that centering the data would be the best approach, since my initial impression was that
offensive capabilities follow something like a normalized curve, where ~70% of players
or more would be "average class" and the remaining would be split between "below average" and
"above average" classes).

Other thoughts: this may overly penalize players who play in years with major superstars (because
the highs are so high)
"""
#now to graph

# choosing kmeans++ as initializer after trying the random start first
y_pred = KMeans(n_clusters = 3).fit_predict(X_test_max_abs_05)
# plot eigenvectors with assumed number of clusters (3 based on the initial,
#good, average, and bad assumption)
plt.scatter(X_reduced2[:, 0], X_reduced2[:, 1], c = y_pred, s = 5, marker = "x")

plt.title("MaxAbs Scaler: KMeans clustering")
#first eigenvector
plt.xlabel("1st PCA component")
#second eigenvector
plt.ylabel("2nd PCA component")
plt.show()

"""
Based on the plots, the best (visual) separation occurs with 2-4 clusters,
which is within the "that makes sense" range of possibility.

However, I wanted to try some of the other clustering options to see if there
were different results with other methods. When using MeanShift to estimate
the number of clusters, we got 2 clusters. However, upon looking at the plot,
the actual visual clustering was poor, which is not surprising considering
that the method is best used to discover blobs in a smooth density of samples,
which my set is... not.
The other clustering method I wanted to try was affinity propagation, in case
there was an "exemplar" type player of a subclass. However, based on that model,
there were 26 clusters... which was too many to have any real meaning to me.
"""
#compute meanshift
"""
    # do meanshift
    bandwidth = estimate_bandwidth(X_test_max_abs_05, quantile = 0.2, n_samples = 390)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    ms.fit(X_test_max_abs_05)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)
    print("# of est clusters: %d" % n_clusters_)

    # plot

    from itertools import cycle

    plt.figure(1)
    plt.clf()
    colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
    for k, col in zip(range(n_clusters_), colors):
        my_members = labels == k
        cluster_center = cluster_centers[k]
        plt.plot(X_reduced2[my_members, 0], X_reduced2[my_members, 1], col + ".")
        plt.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=14,
        )
    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()
    """
#compute affinity propagation
"""
    af = AffinityPropagation(random_state=0).fit(X_test_max_abs_05)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)



    #plot result
    from itertools import cycle

    plt.close("all")
    plt.figure(1)
    plt.clf()

    colors = cycle("bgrcmykbgrcmykbgrcmykbgrcmyk")
    for k, col in zip(range(n_clusters_), colors):
        class_members = labels == k
        cluster_center = X_reduced2[cluster_centers_indices[k]]
        plt.plot(X_reduced2[class_members, 0], X_reduced2[class_members, 1], col + ".")
        plt.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=14,
        )
        for x in X_reduced2[class_members]:
            plt.plot([cluster_center[0], x[0]], [cluster_center[1], x[1]], col)

    plt.title("Estimated number of clusters: %d" % n_clusters_)
    plt.show()
    """



""" Now I will take a look at defensive data and see how good the clustering gets.
I'm expecting this one to be "easier" on account of me assuming it is a binary
question about whether a forward defends or not.

On account of the data being extremely different in nature, I will try all
the scalers again before flattening to 2D.
"""
test = pd.read_csv("rel_forward_defensive_data.csv")
test_2005 = test[test["Season"] == "2005-06"]

# lets do a quick scatter first
"""

    fig4 = plt.figure()
    plt.scatter(test_2005["H/TOI"], test_2005["BS/TOI"])
    plt.title("Blocked Shots/TOI vs Hits/TOI")
    plt.xlabel("H/TOI")
    plt.ylabel("BS/TOI")
    plt.show()
"""

# drop the descriptive columns
test_2005 = test_2005.drop(columns = "Season")
# Then check different scaling methods

#normalization approach to scaling
norm1 = MinMaxScaler().fit(test_2005)
X_n_05 = norm1.transform(test_2005)
#max abs scaler
max_abs1 = MaxAbsScaler().fit(test_2005)
X_max_05 = max_abs1.transform(test_2005)
#standardization approach to scaling
standscale1 = StandardScaler().fit(test_2005)
X_stand_05 = standscale1.transform(test_2005)
#robust scaler
robust1 = RobustScaler().fit(test_2005)
X_rob_05 = robust1.transform(test_2005)
#power transformer scaler
powtrans1 = PowerTransformer().fit(test_2005)
X_pow_05 = powtrans1.transform(test_2005)

pca1 = PCA(n_components = 3)
X_r = pca1.fit(X_n_05).transform(X_n_05)
print("explained variance ratios with normalizer: %s"% str(pca1.explained_variance_ratio_))
#
X_r1 = pca1.fit(X_stand_05).transform(X_stand_05)
print("explained variance ratios with standardizer: %s"% str(pca1.explained_variance_ratio_))
#
X_r2 = pca1.fit(X_max_05).transform(X_max_05)
print("explained variance ratios with max_abs scaler: %s"% str(pca1.explained_variance_ratio_))
#
X_r3 = pca1.fit(X_rob_05).transform(X_rob_05)
print("explained variance ratios with robust scaler: %s"% str(pca1.explained_variance_ratio_))
#
x_r4 = pca1.fit(X_pow_05).transform(X_pow_05)
print("explained variance ratios with power transformer scaler: %s"% str(pca1.explained_variance_ratio_))

# as anticipated, max_ab scaler still performed the best

y_pred1 = KMeans(n_clusters = 2).fit_predict(X_max_05)
# plot eigenvectors with assumed number of clusters (2 for a 1st go)
plt.scatter(X_r2[:, 0], X_r2[:, 1], c = y_pred1, s = 5, marker = "x")

plt.title("MaxAbs Scaler: KMeans clustering with forward defense data")
#first eigenvector
plt.xlabel("1st PCA component")
#second eigenvector
plt.ylabel("2nd PCA component")
plt.show()

"""
Let's do a quick thought experiment. At the moment we have lots of possibilities
based on previously mentioned assumptions:
1. 2 offensive classes and 2 defensive classes
2. 2 offensive classes and 3 defensive classes
3. 3 offensive classes and 2 defensive classes
4. 3 offensive classes and 3 defensive classes
5. 4 offensive classes and 2 defensive classes
6. 4 offensive classes and 3 defensive classes

(There can never be more than 3 defensive classes based on the amount of data
that I previously labeled as defensive, since there are only 3 features).

According to traditional forward performance analysis (previously mentioned in
one of the first comments of this documentation), there are 6 forward classes.
I have no idea whether this is actually the case, but let's move forward as if
it is for exploration purposes (since this is an unsupervised learning project anyway).

Applying basic statistics to our list of class combination possibilities, this
means that we have either:
a. 2 offensive classes and 3 defensive classes
or
b. 3 offensive classes and 2 defensive classes

The latter is far more likely based on my understanding of hockey, so we will
move on with a new assumption:
    The type of forward a player is caused by a combination of their specific
    offensive class and defensive class, for 6 possible combinations. For
    labelling sake, we will say:
    - type1 = Offensive class 1 + Defensive class 1
    - type2 = Offensive class 1 + Defensive class 2
    - type3 = Offensive class 2 + Defensive class 1
    - type4 = Offensive class 2 + Defensive class 2
    - type5 = Offensive class 3 + Defensive class 1
    - type6 = Offensive class 3 + Defensive class 2
"""

#now let's combine the labels
f1 = pd.read_csv("rel_forward_offensive_data_for_pca_wname.csv")
#pull 2005-06 data
f1_2005 = f1[f1["Season"] == "2005-06"]
#get classes as list
off_class_list = y_pred.tolist()
# this is with 3 clusters/classes using the previous assumptions
f1_2005.insert(loc=1,column ="Offense Class", value = off_class_list)
#f1_2005.to_csv("2005_forwards_classified.csv", index = False)

test1 = pd.read_csv("rel_forward_defensive_data_wname.csv")
test1_2005 = test1[test1["Season"] == "2005-06"]

def_class_list = y_pred1.tolist()
#this is with 2 clusters/classes using the previously mentioned assumptions
test1_2005.insert(loc=1, column = "Defense Class", value = def_class_list)

combo_class_2005 = pd.merge(f1_2005, test1_2005, on=["Name", "Season", "+/-"])
#create list of types corresponding to player indexes
type_list = []
for x in range(0, len(combo_class_2005)):
    if combo_class_2005.iat[x, combo_class_2005.columns.get_loc("Offense Class")] == 0:
        if combo_class_2005.iat[x, combo_class_2005.columns.get_loc("Defense Class")] == 0:
            # assign type 1
            type_list.append(1)
        else:
            type_list.append(2)
    elif combo_class_2005.iat[x, combo_class_2005.columns.get_loc("Offense Class")] == 1:
        if combo_class_2005.iat[x, combo_class_2005.columns.get_loc("Defense Class")] == 0:
            type_list.append(3)
        else:
            type_list.append(4)
    # offense class must be 2 (out of 0, 1, 2)
    elif combo_class_2005.iat[x, combo_class_2005.columns.get_loc("Defense Class")] == 0:
        type_list.append(5)
    else:
        type_list.append(6)
#add type list to combination df
combo_class_2005.insert(loc = 1, column = "Forward Class", value = type_list)
#combo_class_2005.to_csv("foward_classes_2005.csv", index = False)

"""
Now from here there is just checking whether these classifications mean anything.

What I will want to check is:
a) Player class over time (especially on entry-level-contract, to see if
   and patterns are common); this is why I started with Ovechkin/Crosby's rookie
   year. (It's also conveniently considered the first year (or close to it) of
   a new era of hockey).
b) Team player composition

The first part is quite important. On account of having as many data points available,
I will run all the years then look at groups of points (i.e. forwards).
Also, I didn't mention this assumption before, but I am also assuming that
defensemen play fundamentally different role/have different skills than forwards.
"""
# f1 is the name of the df storing all the forward offensive data
# test1 is the name of the df storing all the forward defensive data
# there are 14 seasons to consider that need to be scaled separately

"""
df = pd.read_csv("rel_forward_defensive_data_wteam.csv")
df_2005 = df[df["Season"] == "2005-06"]
teams = df_2005["Team"].tolist()
combo_class_2005.insert(loc = 1, column = "Team", value = teams)
#combo_class_2005.to_csv("foward_classes_2005_teams.csv", index = False)
#let's check the top teams first
# note: atlanta thrashers moved and became the winnepeg jets later
# pheonix also became arizona
team_abbrev = ["SJS", "NYR", "WSH", "OTT", "PIT", "CAR", "ATL", "TBL", "ANA",
"NJD", "FLA", "COL", "DET", "NSH", "PHI", "MIN", "VAN", "TOR", "DAL", "EDM",
"BOS", "BUF", "CGY", "NYI", "PHX", "LAK", "MTL", "CBJ", "CHI", "STL"]
team_check1 = combo_class_2005[combo_class_2005["Team"] == "OTT"]
# top regular season teams in 2005-06 were NJD, OTT, CAR, DET, CAL, DAL (division leaders)
# following them (based on points): BUF, NSH, PHI, NYR, SJS, ANA, EDM
# to hold the count of each type
team_for_comp = [0, 0, 0, 0, 0, 0]
comp_labels = ["Type 1", "Type 2", "Type 3", "Type 4", "Type 5", "Type 6"]
for i in range(0, len(team_for_comp)):
    temp = team_check1[team_check1["Forward Class"] == i+1]
    team_for_comp[i] = len(temp)

from tabulate import tabulate
print(tabulate({"Player Types" : comp_labels, "Count" : team_for_comp}, headers = "keys"))
"""
