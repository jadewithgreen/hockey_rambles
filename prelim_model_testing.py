""" want to test the model approach
    will try to first to do it on the forward data points
    expanding the sample set from 390 -> ~ 14*390 = 5460 """

#import modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
import sklearn
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA

# read in csv file with point identifiers
forward_off_data = pd.read_csv("rel_forward_offensive_data_for_pca_wname_team.csv")
forward_def_data = pd.read_csv("rel_forward_defensive_data_wteam.csv")

defenseman_off_data = pd.read_csv("rel_defenseman_offensive_data_for_pca_wname_team.csv")
defenseman_def_data = pd.read_csv("rel_defensemen_defensive_data_wteam.csv")

#pull each individual season for scaling
# first offensive data
#number of seasons for analysis
num_seasons = 12
#instantiate list (of scaled dataframes)
X_off_list = []
X_off_d_list = []
X_def_list = []
X_def_d_list = []

starting_year = "2005"
for i in range(0, num_seasons):
    if int(starting_year[len(starting_year)-2:]) + 1 < 10 :
        season = starting_year + "-0" + str(int(starting_year[len(starting_year)-2:])+1)
    else:
        season = starting_year + "-" + str(int(starting_year[len(starting_year)-2:])+1)
    # pull forward offensive data for the season
    X_off_temp = forward_off_data[forward_off_data["Season"] == season]
    X_off_temp_d = defenseman_off_data[defenseman_off_data["Season"] == season]
    #drop non-numerical columns
    X_off_temp = X_off_temp.drop(columns = ["Season", "Team", "Name"])
    X_off_temp_d = X_off_temp_d.drop(columns = ["Season", "Team", "Name"])
    # get the scaling
    max_ab_o = MaxAbsScaler().fit(X_off_temp)
    max_ab_o_d = MaxAbsScaler().fit(X_off_temp_d)
    # get the scaled data
    X_off_scaled = max_ab_o.transform(X_off_temp)
    X_off_d_scaled = max_ab_o_d.transform(X_off_temp_d)
    #add scaled data to main offense data list
    X_off_list.append(X_off_scaled)
    X_off_d_list.append(X_off_d_scaled)
    # then repeat with defensive data
    # pull forward defensive data for the season
    X_def_temp = forward_def_data[forward_def_data["Season"] == season]
    X_def_temp_d = defenseman_def_data[defenseman_def_data["Season"] == season]
    #drop non-numerical columns
    X_def_temp = X_def_temp.drop(columns = ["Season", "Team", "Name"])
    X_def_temp_d = X_def_temp_d.drop(columns = ["Season", "Team", "Name"])
    # get the scaling
    max_ab_d = MaxAbsScaler().fit(X_def_temp)
    max_ab_d_d = MaxAbsScaler().fit(X_def_temp_d)
    #get the scaled data
    X_def_scaled = max_ab_d.transform(X_def_temp)
    X_def_d_scaled = max_ab_d_d.transform(X_def_temp_d)
    #add scaled data to main defensive data list
    X_def_list.append(X_def_scaled)
    X_def_d_list.append(X_def_d_scaled)
    #bump up starting year
    starting_year = season[:2] + season[len(season)-2:]
#combine the lists of np arrays into singular nd arrays
X_off = np.concatenate(X_off_list)
X_off_d = np.concatenate(X_off_d_list)
X_def = np.concatenate(X_def_list)
X_def_d = np.concatenate(X_def_d_list)

#in case I want the dataframes as well:
X_off_df = pd.DataFrame(X_off, columns = ["ESA/60", "ESG/60", "+/-", "SH%"])
X_off_d_df = pd.DataFrame(X_off_d, columns = ["ESA/60", "ESG/60", "+/-", "SH%"])
X_def_df = pd.DataFrame(X_def, columns = ["H/TOI", "BS/TOI", "+/-"])
X_def_d_df = pd.DataFrame(X_def_d, columns = ["H/TOI", "BS/TOI", "+/-"])


"""
Now that I have my full combined datasets, each subset's features scaled by the
season's maximum absolute value of that feature, I will start PCA.
"""
pca = PCA(n_components = 3)
X_r = pca.fit(X_off).transform(X_off)
print("explained variance ratios in forward offense data: %s"% str(pca.explained_variance_ratio_))
pca1 = PCA(n_components = 2)
X_r1 = pca1.fit(X_def).transform(X_def)
print("explained variance ratios in forward defense data: %s"% str(pca1.explained_variance_ratio_))

pca_d = PCA(n_components = 2)
X_rd = pca_d.fit(X_off_d).transform(X_off_d)
print("explained variance ratios in defenseman offense data: %s"% str(pca.explained_variance_ratio_))
pca_d1 = PCA(n_components = 2)
X_rd1 = pca_d1.fit(X_def_d).transform(X_def_d)
print("explained variance ratios in defenseman defense data: %s"% str(pca.explained_variance_ratio_))

#check kmeans

from sklearn.cluster import KMeans
# checked several seeds, results are pretty stable
y_pred_off = KMeans(n_clusters = 3, random_state = 0).fit_predict(X_off)
y_pred_def = KMeans(n_clusters = 2, random_state = 0).fit_predict(X_def)

y_pred_off_d = KMeans(n_clusters = 2, random_state = 8).fit_predict(X_off_d)
y_pred_def_d = KMeans(n_clusters = 2, random_state = 8).fit_predict(X_def_d)
#plot along eigenvector axes
fig, axs = plt.subplots(2, 2)


axs[0, 0].scatter(X_r[:, 0], X_r[:, 1], c = y_pred_off, s = 5, marker = "x")
axs[0, 0].set_title("KMeans clusters with forward offensive data")
axs[0, 1].scatter(X_r1[:, 0], X_r1[:, 1], c = y_pred_def, s = 5, marker = "o")
axs[0, 1].set_title("KMeans clusters with forward defensive data")
axs[1, 0].scatter(X_rd[:, 0], X_rd[:, 1], c = y_pred_off_d, s = 5, marker = "x")
axs[1, 0].set_title("KMeans clusters with defenseman offensive data")
axs[1, 1].scatter(X_rd1[:, 0], X_rd1[:, 1], c = y_pred_def_d, s = 5, marker = "o")
axs[1, 1].set_title("KMeans clusters with defenseman offensive data")


# comment this in or out to show the plot
#plt.show()

# now combine labels
off_class_list = y_pred_off.tolist()
def_class_list = y_pred_def.tolist()
off_d_class_list = y_pred_off_d.tolist()
def_d_class_list = y_pred_def_d.tolist()
# add back the qualitative labels
X_off_df.insert(loc = 0, column = "Name", value = forward_off_data["Name"].tolist())
X_off_df.insert(loc = 0, column = "Team", value = forward_off_data["Team"].tolist())
X_off_df.insert(loc = 0, column = "Season", value = forward_off_data["Season"].tolist())
X_off_d_df.insert(loc = 0, column = "Name", value = defenseman_off_data["Name"].tolist())
X_off_d_df.insert(loc = 0, column = "Team", value = defenseman_off_data["Team"].tolist())
X_off_d_df.insert(loc = 0, column = "Season", value = defenseman_off_data["Season"].tolist())

# add offensive class as column/feature
X_off_df.insert(loc = 2, column = "Offensive Class", value = off_class_list)
X_off_d_df.insert(loc = 2, column = "Offensive Class", value = off_d_class_list)
#add back qualitative labels
X_def_df.insert(loc = 0, column = "Name", value = forward_def_data["Name"].tolist())
X_def_df.insert(loc = 0, column = "Team", value = forward_def_data["Team"].tolist())
X_def_df.insert(loc = 0, column = "Season", value = forward_def_data["Season"].tolist())
X_def_d_df.insert(loc = 0, column = "Name", value = defenseman_def_data["Name"].tolist())
X_def_d_df.insert(loc = 0, column = "Team", value = defenseman_def_data["Team"].tolist())
X_def_d_df.insert(loc = 0, column = "Season", value = defenseman_def_data["Season"].tolist())
# add defensive class as column/feature
X_def_df.insert(loc = 2, column = "Defensive Class", value = def_class_list)
X_def_d_df.insert(loc = 2, column = "Defensive Class", value = def_d_class_list)
X_full = pd.merge(X_off_df, X_def_df, on=["Name", "Team", "Season", "+/-"])
X_full_d = pd.merge(X_off_d_df, X_def_d_df, on=["Name", "Team", "Season", "+/-"])

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

Using the assumptions above (and then some), we will use the "offensive"
and "defensive" classes to describe a "forward class."
"""
forward_class_list = [] #create empty list to append to
for i in range(0, len(X_full)): #iterate over all the forwards
    if X_full.iat[i, X_full.columns.get_loc("Offensive Class")] == 0:
        if X_full.iat[i, X_full.columns.get_loc("Defensive Class")] == 0:
            # assign type 1
            forward_class_list.append(1) #assign type 1/ FClass 1
        else:
            forward_class_list.append(2) # assign type 2/ FClass 2
    elif X_full.iat[i, X_full.columns.get_loc("Offensive Class")] == 1:
        if X_full.iat[i, X_full.columns.get_loc("Defensive Class")] == 0:
            forward_class_list.append(3) #type 3/ FClass 3
        else:
            forward_class_list.append(4) #type 4/ FClass 4
    # offensive class must be 2 (out of 0, 1, 2)
    elif X_full.iat[i, X_full.columns.get_loc("Defensive Class")] == 0:
        forward_class_list.append(5) #type 5/ FClass 5
    else:
        forward_class_list.append(6) #type 6/ FClass 6
#add type list to combination df
X_full.insert(loc = 1, column = "Forward Class", value = forward_class_list)

"""
We then need to do something similar to achieve the 4 defenseman classes.
"""
defenseman_class_list = [] # create empty list to append to
for i in range(0, len(X_full_d)): #iterate over all the defenseman
    if X_full_d.iat[i, X_full_d.columns.get_loc("Offensive Class")] == 0:
        if X_full_d.iat[i, X_full_d.columns.get_loc("Defensive Class")] == 0:
            defenseman_class_list.append(1) #type 1/ DClass 1
        else:
            defenseman_class_list.append(2) #type 2/ DClass 2
    elif X_full_d.iat[i, X_full_d.columns.get_loc("Defensive Class")] == 0:
        defenseman_class_list.append(3) #type 3/ DClass 3
    else:
        defenseman_class_list.append(4) #type 4/ DClass 4

X_full_d.insert(loc = 1, column = "Defenseman Class", value = defenseman_class_list)

#X_full.to_csv("testf.csv", index = False)
#X_full_d.to_csv("testd.csv", index = False)
""" I checked some players that are known to have been more consistent and some
that are less, and the groupings seem to reflect reality. That being the case,
more important than the player classifications (arguably) are the class spreads
within any given team any given year.
If I can find some patterns there, then we would potentially be able to test
a supervised learning model to predict real values (like the number of points a
team achieve in the regular season, more valuable) or categories (type of team
play needed to be successful/tier of team construction)

In order to check this, I need to construct a new table/data frame
need to sort by season, then sort by team
fraction of total players are class 1, class 2, etc

Most straightforward way is to add 6 features (% forward class for each)
"""

team_data = pd.read_csv("raw team data 2005-2017.csv", encoding = "utf-8")
# pull out list of team abbreviations
# teams = team_data["Team"].unique()
# pull out list of seasons
seasons = team_data["Season"].unique()
# get rosters from compiled file before
rosters = pd.read_csv("rel_skater_data.csv", encoding = "utf-8")
rosters = rosters[["Season", "Name", "Team", "Pos"]]
# sort the team data by season and team (should already be sorted by season)
team_data = team_data.sort_values(["Season", "Team", "PTS", "W", "L", "OT","GF", "GA"],
                            ascending = [True, True, False, False, False, False, False, False])
# cut all the other columns out (which are unformatted)
team_data = team_data[["Season", "Team", "W", "L", "OT", "PTS", "GF", "GA"]]
# set up empty dataframe with the descriptive features I want
# in this case FClass 1 refers to the number of forwards a given team has in
#Class 1 in a given season
class_comp = pd.DataFrame(columns = ["Season", "Team", "FClass 1", "FClass 2", "FClass 3", "FClass 4",
                                        "FClass 5", "FClass 6","DClass 1", "DClass 2", "DClass 3", "DClass 4"])
for i in range(0, num_seasons):
    # pull data from relevant season
    season_data = team_data[team_data["Season"] == seasons[i]]

    # get team list for relevant season
    season_team_list = season_data[["Team"]]
    for j in range(len(season_team_list)):
        # narrow down roster to indicated season
        season_rosters = rosters[rosters["Season"] == seasons[i]]
        # narrow down roster to indicated team
        season_roster = season_rosters[season_rosters["Team"] == season_team_list.iat[j, 0]]
        # pull out group of forwards and group of defenseman
        forwards = season_roster[season_roster["Pos"] == "F"]
        defensemen = season_roster[season_roster["Pos"] == "D"]
        # reminder: named the dataframes X_full and X_full_d for forwards and defensemen
        season_forward_classes = [0, 0, 0, 0, 0, 0]
        for l in range(len(forwards)):
        # pull class data from full matrix in 3 steps
            all_classes = X_full[X_full["Season"] == seasons[i]]
            f_classes = all_classes[all_classes["Team"] == season_team_list.iat[j, 0]]
            player_name = forwards.iat[l, forwards.columns.get_loc("Name")]
            fclass = f_classes.loc[f_classes["Name"] == player_name,"Forward Class"].iloc[0]
            # up the count for the relevant class
            season_forward_classes[fclass-1]+=1
        season_defense_classes = [0, 0, 0, 0]
        # then do the same thing for the defensemen
        for m in range(len(defensemen)):
            # pull class data from full matrix
            all_classes = X_full_d[X_full_d["Season"] == seasons[i]]
            d_classes = all_classes[all_classes["Team"] == season_team_list.iat[j, 0]]
            player_name = defensemen.iat[m, defensemen.columns.get_loc("Name")]
            dclass = d_classes.loc[d_classes["Name"] == player_name, "Defenseman Class"].iloc[0]
            #up the count for the relevant class
            season_defense_classes[dclass-1]+=1
            # getting issue specifically with the Z with the latin accent
            # unicode character in question: U+017D
            # put below in to catch encoding error
            # forgot to put encoding in previous to_csv calls
            # I'll keep this in comments for note to self
            """
            if temp.size > 0:# put this in to catch an encoding error
                dclass = temp.iloc[0]
                season_defense_classes[dclass-1]+=1
            else:
                print (player_name)
                """
        #make all the above skater class information into a dataframe
        #going to approach this by making the previous list into dictionary
        # where keys are the column names and the values are the
        # season's defense classes and season's forward classes
        for_key_list = ["FClass 2", "FClass 3", "FClass 4",
                                                "FClass 5", "FClass 6"]
        for_dict = dict(zip(for_key_list, season_forward_classes))
        def_key_list = ["DClass 1", "DClass 2", "DClass 3", "DClass 4"]
        def_dict = dict(zip(def_key_list, season_defense_classes))
        #then add the season and team information to both
        for_dict["Season"] = seasons[i]
        def_dict["Season"] = seasons[i]
        for_dict["Team"] = season_team_list.iat[j, 0]
        def_dict["Team"] = season_team_list.iat[j, 0]
        # merge the two dictionaries (add the keys for defenseman classes)
        for_dict.update(def_dict)
        player_dict = for_dict
        # now make into dataframe
        class_comp = class_comp.append(player_dict, ignore_index = True)
# check end array, should be 30 (num teams per season) x 12 (num seasons)
#print(class_comp.shape)
# fill all the null values
class_comp = class_comp.fillna(0)
"""
Was curious about some of the distributions
class_comp_2006 = class_comp[class_comp["Season"] == "2006-07"]
#names = ["F1", "F2", "F3", "F4","F5", "F6","D1", "D2", "D3", "D4"]
all_values = class_comp_2006[["FClass 1", "FClass 2", "FClass 3", "FClass 4",
                                        "FClass 5", "FClass 6","DClass 1", "DClass 2", "DClass 3", "DClass 4"]]

total_f1 = all_values["FClass 1"].sum()
total_f2 = all_values["FClass 2"].sum()
total_f3 = all_values["FClass 3"].sum()
total_f4 = all_values["FClass 4"].sum()
total_f5 = all_values["FClass 5"].sum()
total_f6 = all_values["FClass 6"].sum()
total_d1 = all_values["DClass 1"].sum()
total_d2 = all_values["DClass 2"].sum()
total_d3 = all_values["DClass 3"].sum()
total_d4 = all_values["DClass 4"].sum()

print("Total FClass1:" + str(total_f1))
print("Total FClass2:" + str(total_f2))
print("Total FClass3:" + str(total_f3))
print("Total FClass4:" + str(total_f4))
print("Total FClass5:" + str(total_f5))
print("Total FClass6:" + str(total_f6))
print("Total DClass1:" + str(total_d1))
print("Total DClass2:" + str(total_d2))
print("Total DClass3:" + str(total_d3))
print("Total DClass4:" + str(total_d4))
"""

"""Now that we have the dataframe that spells out the team comp for every team
for every season in our (limited) set, we can test the ability to have
some function F(X_c) = R, where X_c is a matrix with the values corresponding
to the composition of the team in a given season and R is some real-world
result. We can test whether there is any correlation that gives us above with
things like :

    - R = number of points in a season (regression)
    - R = Whether the team makes the playoffs (boolean, 2 cluster classification)
    - R = Goals For
    - R = Goals Against

    and so on.

Setting the R for the points in a season is the first test I want to do. The
simplest way to test this is using a linear model, where the class_comp
values defined by the previous classification process are values v_i, and those
are multiplied by weights w to get get R_p, the target number of points.

I will start this by linear regression. I want to start with ordinary
least squares even though the features in this case are not completely
independent.

"""
from sklearn import linear_model
reg = linear_model.LinearRegression()
from sklearn.metrics import mean_squared_error, r2_score
# fit to the first 5  years available
train_season_list = ["2005-06", "2006-07", "2007-08", "2008-09", "2009-10"]
#skipping the lockout year because those values are going to be whack
test_season_list = ["2011-12", "2013-14", "2014-15", "2015-16", "2016-17"]
class_comp_05to10 = class_comp[class_comp["Season"].isin(train_season_list)]
class_comp_10to17 = class_comp[class_comp["Season"].isin(test_season_list)]
X_train = class_comp_05to10.drop(columns = ["Season", "Team"])
X_test = class_comp_10to17.drop(columns = ["Season", "Team"])
print(X_train.shape)
# pull the points values from master df
# this is the same data frame used to format all the data, so all the values
#should be ordered correctly
team_points = team_data[["Season", "Team", "PTS"]]
#take only first 5 years
team_points_05to10 = team_points[team_points["Season"].isin(train_season_list)]
team_points_10to17 = team_points[team_points["Season"].isin(test_season_list)]
team_points_train = team_points_05to10.drop(columns = ["Season", "Team"])
team_points_test = team_points_10to17.drop(columns = ["Season", "Team"])
print(team_points_train.shape)
# train model using 2005-2010 data
reg.fit(X_train, team_points_train)

#make predictions using the test set
y_pred = reg.predict(X_test)
print("coefficients: \n", reg.coef_)
# mean squared error
print("mean squared error: %.2f" % mean_squared_error(team_points_test, y_pred))
# coefficient of determinant; 1 is perfect
print("OLS coefficient of determination for points pred: %.2f" % r2_score(team_points_test, y_pred))

"""
Got 74% accuracy using this, which is actually... surprisingly high considering
how this model got developed.

Next, I want to check the goals for (GF) and goals against (GA) together, since
the input matrix X is a combination of offensive and defensive data.
Working under the assumption here that the features are correlated with one another
since it just seems like your forward types, for example, will control what
kind of defensemen your team is willing to keep.
"""
# new target training and test sets
team_gf_ga = team_data[["Season", "Team", "GF", "GA"]]
team_gf_ga_05to10 = team_gf_ga[team_gf_ga["Season"].isin(train_season_list)]
team_gf_ga_10to17 = team_gf_ga[team_gf_ga["Season"].isin(test_season_list)]
team_gf_ga_train = team_gf_ga_05to10.drop(columns = ["Season", "Team"])
team_gf_ga_test = team_gf_ga_10to17.drop(columns = ["Season", "Team"])
"""
Tried doing this as a set of target variables, but kept on getting negatives
r2_score values.... so gave up on that.
"""
#separate variables
team_gf_train = team_gf_ga_train.drop(columns = "GA")
team_ga_train = team_gf_ga_train.drop(columns = "GF")
team_gf_test = team_gf_ga_test.drop(columns = "GA")
team_ga_test = team_gf_ga_test.drop(columns = "GF")

reg2 = linear_model.LinearRegression()
reg2.fit(X_train, team_gf_train)
reg3 = linear_model.LinearRegression()
reg3.fit(X_train, team_ga_train)
#make predictions
y_pred2 = reg2.predict(X_test)
y_pred3 = reg3.predict(X_test)
print("OLS coefficient of determination for GF pred: %.2f" % r2_score(team_gf_test, y_pred2))
print("OLS coefficient of determination for GA pred: %.2f" % r2_score(team_ga_test, y_pred3))

"""
These results were all also interestingly negative... fascinating.
"""



