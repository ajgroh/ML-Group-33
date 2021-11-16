#ML Project
import pandas as pd
import numpy as np
import csv
####SETTINGS#####
pd.set_option('display.max_colwidth',5000)


####### Write Header of CSV data file ##########
column_names = ["PointsPerGame", "PointsAgainstPerGame", "Off1stDownsPerGame", "1stDownsGivenPerGame", "OffTotalYardsPerGame", "YardsGivenUpPerGame", "OffPassYardsPerGame", "PassYardsGivenUpPerGame", "OffRushYardsPerGame", "RushYardsGivenUpPerGame", "OffTurnoversPerGame", "TurnoversCausedPerGame", "MadePlayoffs"]
file_name = "./Data/NFLData.csv"

with open(file_name, 'w') as csv_file:
    csvwriter = csv.writer(csv_file)
    csvwriter.writerow(column_names)

    #####FILES TO READ##########
    data_files = ["Cardinals2020.csv", "Cardinals2019.csv", "Cardinals2018.csv", "Falcons2020.csv", "Falcons2019.csv", "Falcons2018.csv", "Ravens2020.csv", "Ravens2019.csv", "Ravens2018.csv", "Bills2020.csv", "Bills2019.csv", "Bills2018.csv", "Panthers2020.csv", "Panthers2019.csv", "Panthers2018.csv", "Bears2020.csv", "Bears2019.csv", "Bears2018.csv", "Bengals2020.csv", "Bengals2019.csv", "Bengals2018.csv", "Browns2020.csv", "Browns2019.csv", "Browns2018.csv", "Cowboys2020.csv", "Cowboys2019.csv", "Cowboys2018.csv", "Broncos2020.csv", "Broncos2019.csv", "Broncos2018.csv", "Lions2020.csv", "Lions2019.csv", "Lions2018.csv", "Packers2020.csv", "Packers2019.csv", "Packers2018.csv", "Texans2020.csv", "Texans2019.csv", "Texans2018.csv", "Colts2020.csv", "Colts2019.csv", "Colts2018.csv", "Jaguars2020.csv", "Jaguars2019.csv", "Jaguars2018.csv", "Chiefs2020.csv", "Chiefs2019.csv", "Chiefs2018.csv", "Raiders2020.csv", "Raiders2019.csv", "Raiders2018.csv", "Chargers2020.csv", "Chargers2019.csv", "Chargers2018.csv", "Dolphins2020.csv", "Dolphins2019.csv", "Dolphins2018.csv", "Vikings2020.csv", "Vikings2019.csv","Vikings2018.csv", "Patriots2020.csv", "Patriots2019.csv", "Patriots2018.csv", "Saints2020.csv", "Saints2019.csv", "Saints2018.csv", "Giants2020.csv", "Giants2019.csv", "Giants2018.csv", "Jets2020.csv", "Jets2019.csv", "Jets2018.csv", "Eagles2020.csv", "Eagles2019.csv", "Eagles2018.csv", "Stealers2020.csv", "Stealers2019.csv", "Stealers2018.csv", "sf2020.csv", "sf2019.csv", "sf2018.csv", "Seahawks2020.csv", "Seahawks2019.csv", "Seahawks2018.csv", "Bucs2020.csv", "Bucs2019.csv", "Bucs2018.csv", "Titans2020.csv", "Titans2019.csv", "Titans2018.csv", "Wash2020.csv", "Wash2019.csv", "Wash2018.csv", "Rams2020.csv", "Rams2019.csv", "Rams2018.csv"]
    # 0 = didnt make playoffs, 1 = made playoffs
    made_playoffs_data = ["0", "0", "0", "0", "0", "0", "1", "1", "1", "1", "1", "0", "0", "0", "0", "1", "0", "1", "0", "0", "0", "1", "0", "0", "0",
    "0", "1", "0", "0", "0", "0", "0", "0", "1", "1", "0", "0", "1", "1", "1", "0", "1", "0", "0", "0", "1", "1", "1", "0", "0", "0", "0",
    "0", "1", "0", "0", "0", "0", "1", "0", "0", "1", "1", "1", "1", "1", "0", "0", "0", "0", "0", "0", "0", "1", "1", "1", "0", "0",
    "0", "1", "0", "1", "1", "1", "1", "0", "0", "1", "1", "0", "1", "0", "0", "1", "0", "1"]

    #######OPEN AND EDIT FILES##########
    i = 0 #to index through label data
    for file in data_files:
        df = pd.read_csv("./Data/" + file)

        #removes blank first row
        new_header = df.iloc[0] #grab the first row for the header
        df = df[1:] #take the data less the header row
        df.columns = new_header



        df = df.loc[:,["Tm", "Opp", "1stD", "TotYd", "PassY", "RushY", "TO"]]
        df = df.iloc[0:9, :] #get the first 9 weeks

        #removes the bye week row, if present
        for idx, row in df.iterrows():

            if str(row["Opp"][0]) == "Bye Week":
                df = df.drop(idx)
        #fills all nan values with 0
        df = df.fillna(0)
        # print(df.to_numpy())
        data = df.to_numpy()
        off_total_scored = 0; def_total_scored = 0; off_total_1std = 0; def_total_1std = 0; off_total_totyd = 0; def_total_totyd = 0; off_total_passyd = 0; def_total_passyd = 0; off_total_rushyd = 0; def_total_rushyd = 0; off_total_to = 0; def_total_to = 0
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


        row = [off_avg_scored, def_avg_scored, off_avg_1std, def_avg_1std, off_avg_totyd, def_avg_totyd, off_avg_passyd, def_avg_passyd, off_avg_rushyd, def_avg_rushyd, off_avg_to, def_avg_to, made_playoffs_data[i]]
        csvwriter.writerow(row)
        i = i + 1
    # csvwriter.close()



###### Now Data is setup #####
df = pd.read_csv(file_name)
df = df.sample(frac=1)
print(df.to_numpy())
