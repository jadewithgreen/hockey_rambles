# clean player data
""" python3
Objective of this code is to generate revelant player lists (so to eliminate
any player from the data set who did not play a "significant" amount of time).
This does not include goalies, as the complilation data does not include
save percentages.
I chose the cut-off based on some histograms done on 2005 data, as I assume
the time share of 4th liners has increased since then as the league has gotten
younger. (Rookies, even promising ones given 1st line minutes generally are given
ice-time, as hockey can still be quite a seniority/veteran-driven sport).
"""
import pandas as pd
import numpy as np
from decimal import Decimal
#read compiled player data in df
player_data = pd.read_csv("compiled_player_data.csv")
#pull out forwards
forwards_data = player_data.loc[player_data["Pos"] == "F"]
#pull out defensemen
defensemen_data = player_data.loc[player_data["Pos"] == "D"]

# first get rid of the null records for player TOI
forwards_data = forwards_data.loc[forwards_data.TOI.notnull(), :]
defensemen_data = defensemen_data.loc[defensemen_data.TOI.notnull(),:]
# then turn TOI into seconds
for a in range(0, len(forwards_data)):
    #pull TOI value from df
    toi_val = forwards_data.iloc[a]["TOI"]
    #split string by colon then convert into ints
    #multiply ints out to get seconds
    secs = sum(int(x)*60**i for i, x in enumerate(reversed(toi_val.split(":"))))
    forwards_data.iat[a, forwards_data.columns.get_loc("TOI")] = secs

for b in range(0, len(defensemen_data)):
    #pull TOI value from df
    toi_val = defensemen_data.iloc[b]["TOI"]
    #split string and convert
    secs = sum(int(x)*60**i for i,x in enumerate(reversed(toi_val.split(":"))))
    #set TOI value to integer seconds
    defensemen_data.iat[b, defensemen_data.columns.get_loc("TOI")] = secs


#then cut the forwards who didn't play over the previously mentioned time threshold (592s)
#and defensemen that didn't play over 653s (based on defensemen toi 2005 histogram with 5 bins)


rel_forward_data = forwards_data.loc[forwards_data["TOI"] > 592]
rel_defenseman_data = defensemen_data.loc[defensemen_data["TOI"] > 653]
#compiled relevant skaters (non-goalies)
rel_skater_data = rel_forward_data.append(rel_defenseman_data)
rel_skater_data = rel_skater_data.sort_values(["Season", "Team"], ascending = [True, True])
# send this to csv for later use on roster
rel_skater_data.to_csv("rel_skater_data.csv", index = False)
# error free
#and then I have to turn shot percentage into decimals

for c in range(0, len(rel_forward_data)):
    #pull shooting percentage from df
    shp_val = rel_forward_data.iloc[c]["SH%"]
    #chop off end and convert to int
    p = Decimal(shp_val[:len(shp_val)-1])
    #set df cell value to that int
    rel_forward_data.iat[c, rel_forward_data.columns.get_loc("SH%")] = p

for d in range(0, len(rel_defenseman_data)):
    #pull shooting percentage from df
    shp_val = rel_defenseman_data.iloc[d]["SH%"]
    #chop off end and convert to int
    p = Decimal(shp_val[:len(shp_val)-1])
    #set df cell value to that int
    rel_defenseman_data.iat[d, rel_defenseman_data.columns.get_loc("SH%")] = p

""" I now want to prepare this data for PCA and select for (hopefully) non-redundant
features.

The thought process was, based on my background knowledge, I can skip narrowing
down the feature selection by machine and instead do it manually. A lot of the
recorded features in the data are based on one another (e.g. Points (P) = Goals (G)
+ Assists(A) or Points/Game Player (P/GP) = P/Games Played (GP)). Derivative
features have the same amount of variance (so can't take a Variance Threshold
approach), so this was the easiest for me at my current skill level.

The features I thought were most important to feed into PCA for player offense
analysis were:

ESA/60 (Even Strength Assists/60 minutes)
ESG/60 (Even Strength Goals/60 minutes)
Plus/Minus (Goal Differential when Player is on Ice)
Sh% (Shot % = Number of Shots that get into the goal)

using the following assumptions:

Assumption 1: Special teams (so powerplay/penalty kill, or extremely offense-based
/defense-based game states) sway the "base data" of the player too much and
have more to do with other factors than player offensive capabilities.
--> ESA/60 & ESG/60 data eliminates worry for special teams since this is only
    pulling data from the even-strength (non-special team) game state.
Assumption 2: Plus/Minus can be considered an individual player's offensive stat.
--> Since this is the goal differential when the player is on the ice, 3rd and
    4th liners, which are more likely to be defensive players set against the
    opposing team's 1st line, may be forcefully separated on this stat alone.
Assumption 3: Sh% says something about a player's offensive capability (shot speed,
etc.)
--> A lot of top 6 (1st and 2nd line) players get way more time on ice than the
    bottom 6. Instead of using a stat that is highly dependent on time on ice,
    I wanted to try using a stat that is normally considered to be a "luck"
    stat.
--> A possible alternative would be to take the total shots/(average TOI*GP) for
    shots per time on ice, however, this seems more biased against defensive
    forwards, who are much more likely to be on the penalty kill (a defensive
    game state that doesn't allow for many shots on the opponent, if any)
"""

#pull out relevant offensive features
rel_offensive_for_data = rel_forward_data[["Season","Name", "Team","ESA/60", "ESG/60", "+/-", "SH%"]]
rel_offensive_def_data = rel_defenseman_data[["Season", "Name", "Team","ESA/60", "ESG/60", "+/-", "SH%"]]
#send to csv
rel_offensive_for_data.to_csv("rel_forward_offensive_data_for_pca_wname_team.csv", index = False, encoding = "utf-8")
rel_offensive_def_data.to_csv("rel_defenseman_offensive_data_for_pca_wname_team.csv", index = False, encoding = "utf-8")

# For defensive data, it's a completely different ballgame.

"""

Unfortunately, there isn't a lot of good, clearly defensive data recorded for any
players. Forwards who are more defensive are more likely to be put in defensive
situations (like the penalty kill), which would be reflected in the average time
played short-handed (SH), but ultimately, that's a matter of coach discretion
and would potentially bias the model too much toward how players are already used
(which can be easily found by following that team specifically).

The only stats that could potentially point me to defense are:

- Hits/game (?)
- Blocked Shots/game
- +/- (again)

using the following assumptions:
1. Hits are not just a matter of aggression, and they can be used to slow down
opposing offense (i.e. is a defensive manuever).
    --> Since defenseman typically have
    significantly more hits and forwards, this seemed like a fair assumption to make.
2. Blocked shots are not only performed in defensive game states (like the penalty kill).
    --> If this is not the case, then this number would be highly biased toward
    penalty killers (which, again, is a coach-choice)
3. +/- says something about the player's defensive capability.
    --> I have no idea whether this is the case, but I needed a 3rd feature to try.
"""
# first need to add this data to the forward df
hits_per_TOI = rel_forward_data["HITS"]/rel_forward_data["TOI"]
rel_forward_data.insert(loc = 21, column = "H/TOI", value = hits_per_TOI)
blocks_per_TOI = rel_forward_data["BS"]/rel_forward_data["TOI"]
rel_forward_data.insert(loc = 23, column = "BS/TOI", value = blocks_per_TOI)
# then add to the defenseman df
hits_per_TOI1 = rel_defenseman_data["HITS"]/rel_defenseman_data["TOI"]
rel_defenseman_data.insert(loc = 21, column = "H/TOI", value = hits_per_TOI1)
blocks_per_TOI1 = rel_defenseman_data["BS"]/rel_defenseman_data["TOI"]
rel_defenseman_data.insert(loc = 23, column = "BS/TOI", value = blocks_per_TOI1)
# then take the relevant columns to csv

rel_forward_defensive_data = rel_forward_data[["Season", "Name", "Team", "H/TOI", "BS/TOI", "+/-"]]
rel_forward_defensive_data.to_csv("rel_forward_defensive_data_wteam.csv", index = False, encoding = "utf-8")
rel_defenseman_defensive_data = rel_defenseman_data[["Season", "Name", "Team", "H/TOI", "BS/TOI", "+/-"]]
rel_defenseman_defensive_data.to_csv("rel_defensemen_defensive_data_wteam.csv", index = False, encoding = "utf-8")
