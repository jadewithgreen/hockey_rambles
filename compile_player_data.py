# want to clean and format the quanthockey player data
"""
written in python3
Purpose: To combine datasets over the different seasons and clean existing
dataset.
Results in excel file (also in this repo) of compliled player data.
"""

import numpy as np
import pandas as pd



"""
I used network analyzer in browser(Firefox) and observed the page mentioned
in the GET function. Then I scraped the 14 pages by iterating the relevant
pieces. I used 14*50 = 700 players because there were 30 teams in the NHL
between 2005 and 2017, and the roster is 22 to 23 players. 30*22 = 660, which
is rounded up to the nearest multiple of 50 (# of players on each webpage).

"""

seasons = ['2005-06','2006-07','2007-08','2008-09','2009-10','2010-11',
'2011-12','2012-13','2013-14','2014-15','2015-16','2016-17']
main_list = []
for a in range(0, len(seasons)):
    temp_df_list = []
    for b in range(1, 15):
        string_start = "https://www.quanthockey.com/scripts/AjaxPaginate.php?cat=Season&pos=Players&SS="
        string_a = seasons[a]+"&af=0&nat="+seasons[a]
        string_mid = "&st=reg&sort=P&so=DESC&page="
        string_b = str(b)
        string_end = "&league=NHL&lang=en&rnd=190723706&dt=2&sd=undefined&ed=undefined"
        url_wanted = string_start + string_a + string_mid + string_b + string_end
        temp_df = pd.read_html(url_wanted, header=1, index_col=0)
        temp_df_list.append(temp_df[0])
    temp_p_data_df = pd.concat(temp_df_list)
    season_name = seasons[a]
    new_column = [season_name]*700
    temp_p_data_df.insert(1, "Season", new_column, True)
    main_list.append(temp_p_data_df)
player_data = pd.concat(main_list)
print(player_data.shape)
player_data.to_csv('compiled_player_data.csv', index = False, encoding = "utf-8")
