# practice scrape and clean method

""" Usage:python3
Purpose: Playing around with histograms and data to determine some of the cut-off values used in other modeling. 

"""
import pandas as pd
import numpy as np
import datetime

player_data_df = pd.read_csv("compiled_player_data.csv")

#take 2005-6 season out of player data compilation
player_data_2005 = player_data_df[player_data_df["Season"] == '2005-06']
#pull points column from dataframe
points_2005 = player_data_2005[["P"]]
# created a histogram with 10, 20, and 25 bins
# 20 bins had the greatest separation

#ax = points_2005.plot.hist("P", bins=25)
#fig = ax.get_figure()
#fig.savefig('test_hist.pdf')

min = points_2005.min()
max = points_2005.max()
cutoff = min + int((max - min)/25)*2
print(cutoff)

data_2005 = player_data_2005[["Name","Age","Pos","GP","G","A","P","TOI", "ES","P/GP", "ESP/60"]]
"""
Originally put in this section to only fill nulls for TOI for 0s, but
for this data set, all nulls can be replaced with 0. Assumption for TOI here
is that TOIs that are not recorded are those of non NHL regulars, so even if
the data existed, we would expect to find that the average TOI per game would
be extremely low.
# previous code (used dict because usual method was splicing df):
# rel_data_2005 = data_2005.fillna({"TOI":0})
"""
rel_data_2005 = data_2005.fillna(0)
# turn TOI to seconds
for a in range (0, len(data_2005)):
    #pull TOI value from df
    toi_val = rel_data_2005.iloc[a]["TOI"]
    # check whether the value was one of the previously null values
    is_str = isinstance(toi_val, str)
    # if not former null value (i.e. string date) transform into seconds
    if (is_str):
        secs = sum(int(x)*60**i for i, x in enumerate(reversed(toi_val.split(':'))))
        rel_data_2005.loc[a, "TOI"] = secs

# Turn even-strength TOI (ES) to seconds
for b in range (0, len(data_2005)):
    #pull TOI value from df
    es_val = rel_data_2005.iloc[b]["ES"]
    # check whether the value was one of the previously null values
    is_str = isinstance(es_val, str)
    # if not former null value (i.e. string date) transform into seconds
    if (is_str):
        secs = sum(int(x)*60**i for i, x in enumerate(reversed(es_val.split(':'))))
        rel_data_2005.loc[b, "ES"] = secs

"""
Decided to focus on forwards first here. (Where points are arguably the most
important number)
"""
forwards_2005 = rel_data_2005.loc[rel_data_2005["Pos"] == "F"]
defense_2005 = rel_data_2005.loc[rel_data_2005["Pos"] == "D"]
print(forwards_2005.shape)
"""let's take a look at the histogram quickly
# using points per game (p/gp) here instead of points to account for major injuries/
# or stretches of missed games I don't know about; This of course also
# does weigh more heavily on one-game wonders, but my take is that that is less
# common than injury of seasoned NHLers.
"""
forwards_adj_points_2005 = forwards_2005[["P/GP"]]
#ax1 = forwards_adj_points_2005.plot.hist("P/GP", bins=20)
#fig1 = ax1.get_figure()
#fig1.savefig('player_hist.pdf')
""" Ran for 5, 10, and 20 bins and found drastic differences. The difference
between the 1st(lowest value) and 2nd bin for the 5 bin and 10 bin histogram
was minimal. Decided to go with the 20 bin to cut out the bottom outliers. The
bottom 20th performance (p/gp in this case) occurs at less than half the
frequency of the next bin. Th
"""

min_pgp = forwards_adj_points_2005.min()
max_pgp = forwards_adj_points_2005.max()
pgp_cutoff = min_pgp + int((max_pgp - min_pgp)/20)
#print(pgp_cutoff)

"""
After this, I realized that TOI is a better indicator.
"""
forwards_toi_2005 = forwards_2005["TOI"]
defense_toi_2005 = defense_2005["TOI"]
ax2 = defense_toi_2005.plot.hist("TOI", bins = 5)
fig2 = ax2.get_figure()
fig2.savefig('defense_toi_hist.pdf')

# ran 5 and 10 bins; with 5 bins, first (lowest) bin stood out significantly
min_toi = forwards_toi_2005.min()
max_toi = forwards_toi_2005.max()
avg_toi = forwards_toi_2005.mean()

min_d_toi = defense_toi_2005.min()
max_d_toi = defense_toi_2005.max()
avg_d_toi = defense_toi_2005.mean()

toi_cutoff = min_toi + int((max_toi - min_toi)/5)
toi_d_cutoff = min_d_toi + int((max_d_toi - min_d_toi)/5)
print(toi_d_cutoff)
""" This cutoff ends up being 275 seconds, or almost 5 minutes. This is
less than half of average 4th line minutes in the NHL today, which is ~12 min.
The 5 minute cutoff only cuts off 2 players.
As such, we're going to go with a higher cutoff to ensure we get something
closer to 4 lines x 3 players x 30 teams = 360 players + 30 reserve (13th
forward).
Let's try 10 min/600s, which allows for some below average numbers. There is
no historic data around 4th line minutes, so I'm testing the results of this
assumption (players that play less than 10 minutes a game aren't very relevant).
Random call-ups, for example, might play over that amount for a single game,
but likely not consistently. Putting a 600s threshold gets us 387 players.
Then I manually adjusted the threshold downward until we had 390 players left.
Note that this doesn't guarantee an even 13 players per team, this just accounts
for the total numbers.
"""

rel_forwards_2005 = forwards_2005.loc[forwards_2005["TOI"] > 592]
print(rel_forwards_2005.shape)

""" Since it's well-known
that first liners receive way more minutes than fourth liners, I wanted to take
a points/time measure, like ESP/60 (even strength points/60 minutes).

Let's take a look at a histogram to see if my assumption is correct.
"""

# take even strength points/time df slice
forwards_es_2005 = rel_forwards_2005["ESP/60"]
print(forwards_es_2005.shape)

#remove the forwards that had no even strength points
#rel_forwards_es_2005 = forwards_es_2005[forwards_es_2005["ESP/60"] != 0]
ax3 = forwards_es_2005.plot.hist("ESP/60", bins = 20)
fig3 = ax3.get_figure()
fig3.savefig('ESPper60_test_hist.pdf')
