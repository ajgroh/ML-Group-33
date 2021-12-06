import pandas as pd
import numpy as np
import csv

pd.set_option('display.max_colwidth', 5000)

####### Write Header of CSV data file ##########
column_names = ["PointsPerGame", "PointsAgainstPerGame", "Off1stDownsPerGame", "1stDownsGivenPerGame", "OffTotalYardsPerGame", "YardsGivenUpPerGame",
                "OffPassYardsPerGame", "PassYardsGivenUpPerGame", "OffRushYardsPerGame", "RushYardsGivenUpPerGame", "OffTurnoversPerGame", "TurnoversCausedPerGame", "MadePlayoffs"]
file_name = "./Data/MoreNFLData.csv"

with open(file_name, 'w', newline="") as csv_file:
    csvwriter = csv.writer(csv_file)
    csvwriter.writerow(column_names)

    #####FILES TO READ##########
    teams = ["nwe", "buf", "mia", "nyj", "oti", "clt", "htx", "jax", "rav", "cin", "pit", "cle", "kan", "sdg", "den",
             "rai", "dal", "was", "phi", "nyg", "tam", "atl", "car", "nor", "gnb", "min", "chi", "det", "crd", "ram", "sfo", "sea"]
    years = ["2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009",
             "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020"]
    # 0 = didnt make playoffs, 1 = made playoffs
    made_playoffs_data = []

    #######OPEN AND EDIT FILES##########
    i = 0  # to index through label data
    for team in teams:
        for year in years:
            url = "https://www.pro-football-reference.com/teams/" + \
                team + "/" + year + ".htm#games"
            try:
                df = pd.read_html(url)[1]
            except:
                print(url, " does not exist.")
                continue

            df.columns = df.columns.droplevel()

            if df['Date'].str.contains('Playoffs').any():
                made_playoffs = 1
            else:
                made_playoffs = 0

            df = df.loc[:, ["Tm", "Opp", "1stD",
                            "TotYd", "PassY", "RushY", "TO"]]

            # removes the bye week row, if present
            for idx, row in df.iterrows():

                if str(row["Opp"][0]) == "Bye Week":
                    df = df.drop(idx)

            df = df.iloc[0:9, :]  # get the first 9 weeks
            # fills all nan values with 0
            df = df.fillna(0)

            data = df.to_numpy()
            off_total_scored = 0
            def_total_scored = 0
            off_total_1std = 0
            def_total_1std = 0
            off_total_totyd = 0
            def_total_totyd = 0
            off_total_passyd = 0
            def_total_passyd = 0
            off_total_rushyd = 0
            def_total_rushyd = 0
            off_total_to = 0
            def_total_to = 0
            for game in data:
                off_total_scored += int(game[0])
                def_total_scored += int(game[2])
                off_total_1std += int(game[3])
                def_total_1std += int(game[4])
                off_total_totyd += int(game[5])
                def_total_totyd += int(game[6])
                off_total_passyd += int(game[7])
                def_total_passyd += int(game[8])
                off_total_rushyd += int(game[9])
                def_total_rushyd += int(game[10])
                off_total_to += int(game[11])
                def_total_to += int(game[12])

            game_count = len(data)
            off_avg_scored = off_total_scored/game_count
            def_avg_scored = def_total_scored/game_count
            off_avg_1std = off_total_1std/game_count
            def_avg_1std = def_total_1std/game_count
            off_avg_totyd = off_total_totyd/game_count
            def_avg_totyd = def_total_totyd/game_count
            off_avg_passyd = off_total_passyd/game_count
            def_avg_passyd = def_total_passyd/game_count
            off_avg_rushyd = off_total_rushyd/game_count
            def_avg_rushyd = def_total_rushyd/game_count
            off_avg_to = off_total_to/game_count
            def_avg_to = def_total_to/game_count

            row = [off_avg_scored, def_avg_scored, off_avg_1std, def_avg_1std, off_avg_totyd, def_avg_totyd,
                   off_avg_passyd, def_avg_passyd, off_avg_rushyd, def_avg_rushyd, off_avg_to, def_avg_to, made_playoffs]
            csvwriter.writerow(row)
        # csvwriter.close()


###### Now Data is setup #####
df = pd.read_csv(file_name)
df = df.sample(frac=1)
print(df.to_numpy())
