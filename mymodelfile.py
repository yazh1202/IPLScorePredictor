import pandas as pd
import sklearn as sk
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
from sklearn.metrics import r2_score


class MyModel:
    model = None
    venueMean = float(45.44489566204394)
    teamMean = float(44.84651337796224)
    averageStrikeRate = float(123.92799513109729)

    def __init__(self):
        # Default Case

        MyModel.venueAvg["Average"] = MyModel.venueMean
        MyModel.venueAvg.setdefault("Average")
        # Default Case

        MyModel.teamAverages["Average"] = MyModel.teamMean
        MyModel.teamAverages.setdefault("Average")
        # Default Case

        MyModel.battersData["Average"] = MyModel.averageStrikeRate
        MyModel.battersData.setdefault("Average")
        MyModel.model = LinearRegression()

    def fit(self, training_data):
        try:
            MyModel.model = LinearRegression()
            ballData = training_data[0]
            matchData = training_data[1]
            scoresData = MyModel.makeTrainingData(ballData, matchData)
            MyModel.process(scoresData)
        except:
            return self
        return self

    def predict(self, test_data):
        try:
            testdf = pd.DataFrame(
                columns=["venueAvg", "TeamBattingAvg", "Batter1SR", "Batter2SR", "Batter3SR"])
            for j in range(0, len(test_data)):
                try:
                    # Venue and Team Averages
                    venue = test_data.loc[j, "venue"].strip()[:20]
                    battingTeam = test_data.loc[j, "batting_team"]
                    batterString = (test_data.loc[j, "batsmen"])
                    batters = list(getBatters(batterString))
                    testdf.loc[j, "venueAvg"] = MyModel.MyModel.venueAvg[venue]
                    testdf.loc[j, "TeamBattingAvg"] = MyModel.teamAverages[battingTeam]
                    # BatterStrikeRates
                    testdf.loc[j, 'Batter1SR'] = MyModel.battersData.get(batters[0]) if MyModel.battersData.get(
                        batters[0]) is not None else MyModel.averageStrikeRate
                    testdf.loc[j, 'Batter2SR'] = MyModel.battersData.get(batters[1]) if MyModel.battersData.get(
                        batters[1]) is not None else MyModel.averageStrikeRate
                    testdf.loc[j, 'Batter3SR'] = MyModel.battersData.get(batters[2]) if len(
                        batters) > 2 and MyModel.battersData.get(batters[2]) is not None else MyModel.averageStrikeRate
                except Exception as e:
                    testdf.loc[j, "TeamBattingAvg"] = MyModel.teamAverages[battingTeam]
                    testdf.loc[j, "venueAvg"] = MyModel.venueAvg[venue]
                    testdf.loc[j, 'Batter1SR'] = MyModel.averageStrikeRate
                    testdf.loc[j, 'Batter2SR'] = MyModel.averageStrikeRate
                    testdf.loc[j, 'Batter3SR'] = MyModel.averageStrikeRate
            # print(testdf)
            matchPred = MyModel.model.predict(testdf)
            # print(matchPred)
            return matchPred
        except:
            tempFrame = pd.DataFrame()
            tempFrame.loc[0] = 46
            tempFrame.loc[1] = 47

            return tempFrame

    @staticmethod
    def process(scoresData):
        X = scoresData[["venueAvg", "TeamBattingAvg",
                        "Batter1SR", "Batter2SR", "Batter3SR"]]
        y = scoresData[["RunsScored"]]

        # Train Test Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42)

        # creating a regression model

        MyModel.model = LinearRegression()

        # fitting the model
        MyModel.model.fit(X_train, y_train)

        # making predictions

        predictions = MyModel.model.predict(X_train)

    @staticmethod
    def makeTrainingData(ballData, matchData):

        scoresData = pd.DataFrame(columns=["MatchNo", "MatchID", "Year", "Venue", "Innings", "venueAvg", "TeamBatting", "TeamBowling",
                                  "TeamBattingAvg", "Batters", "Bowlers", "Batter1SR", "Batter2SR", "Batter3SR", "RunsScored", "Wickets"])
        j = int(0)
        mNo = int(0)
        while j < len(matchData):
            # 1st Innings
            matchID = int(matchData.loc[j, "ID"])
            team1 = matchData.loc[j, "Team1"]
            team2 = matchData.loc[j, "Team2"]
            teams = list([team1, team2])
            matchVenue = matchData.loc[j, "Venue"][0:20]
            isPlayed = matchData.loc[j, "WinningTeam"]
            season = matchData.loc[j, "Season"]

            scoresData.loc[j, "MatchNo"] = mNo
            scoresData.loc[j, "MatchID"] = matchID
            scoresData.loc[j, "Year"] = matchData.loc[j, "Season"]
            scoresData.loc[j, "Venue"] = matchVenue
            scoresData.loc[j, "Innings"] = 1
            scoresData.loc[j, "venueAvg"] = MyModel.venueAvg[matchVenue]
            scoresData.loc[j, "TeamBatting"] = teams[0]
            scoresData.loc[j, "TeamBowling"] = teams[1]
            scoresData.loc[j,
                           "TeamBattingAvg"] = MyModel.teamAverages[teams[0]]

            try:
                # Batter and Bowler Details

                batters = list(ballData["batter"][(ballData["ID"] == matchID) & (
                    ballData["overs"] < 6) & (ballData["innings"] == 1)].drop_duplicates())
                scoresData.loc[j, "Batters"] = batters
                bowlers = list(ballData["bowler"][(ballData["ID"] == matchID) & (
                    ballData["overs"] < 6) & (ballData["innings"] == 1)].drop_duplicates())
                scoresData.loc[j, "Bowlers"] = bowlers

                # BatterStrikeRates
                scoresData.loc[j, 'Batter1SR'] = MyModel.battersData.get(
                    batters[0]) if MyModel.battersData.get(batters[0]) is not None else MyModel.averageStrikeRate
                scoresData.loc[j, 'Batter2SR'] = MyModel.battersData.get(
                    batters[1]) if MyModel.battersData.get(batters[1]) is not None else MyModel.averageStrikeRate
                scoresData.loc[j, 'Batter3SR'] = MyModel.battersData.get(batters[2]) if len(
                    batters) > 2 and MyModel.battersData.get(batters[2]) is not None else MyModel.averageStrikeRate
            except Exception as e:
                # Default Values for things
                scoresData.loc[j, "Batters"] = list()
                scoresData.loc[j, "Bowlers"] = list()
                scoresData.loc[j, 'Batter1SR'] = MyModel.averageStrikeRate
                scoresData.loc[j, 'Batter2SR'] = MyModel.averageStrikeRate
                scoresData.loc[j, 'Batter3SR'] = MyModel.averageStrikeRate

            # Runs and Wickets for the team

            scoresData.loc[j, "RunsScored"] = ballData["total_run"][(ballData["ID"] == matchID) & (
                ballData["overs"] < 6) & (ballData["innings"] == 1)].sum()
            scoresData.loc[j, "Wickets"] = ballData["isWicketDelivery"][(
                ballData["ID"] == matchID) & (ballData["overs"] < 6) & (ballData["innings"] == 1)].sum()

            # 2nd Innings
            j += 1
            scoresData.loc[j, "MatchNo"] = mNo
            scoresData.loc[j, "MatchID"] = matchID
            scoresData.loc[j, "Year"] = season
            scoresData.loc[j, "Venue"] = matchVenue
            scoresData.loc[j, "Innings"] = 2
            scoresData.loc[j, "venueAvg"] = MyModel.venueAvg[matchVenue]
            scoresData.loc[j, "TeamBatting"] = teams[1]
            scoresData.loc[j, "TeamBowling"] = teams[0]
            scoresData.loc[j,
                           "TeamBattingAvg"] = MyModel.teamAverages[teams[1]]

            try:
                # Batter and Bowler Details

                batters = list(ballData["batter"][(ballData["ID"] == matchID) & (
                    ballData["overs"] < 6) & (ballData["innings"] == 2)].drop_duplicates())
                scoresData.loc[j, "Batters"] = batters
                bowlers = list(ballData["bowler"][(ballData["ID"] == matchID) & (
                    ballData["overs"] < 6) & (ballData["innings"] == 2)].drop_duplicates())
                scoresData.loc[j, "Bowlers"] = bowlers

                # BatterStrikeRates

                scoresData.loc[j, 'Batter1SR'] = MyModel.battersData.get(
                    batters[0]) if MyModel.battersData.get(batters[0]) is not None else MyModel.averageStrikeRate
                scoresData.loc[j, 'Batter2SR'] = MyModel.battersData.get(
                    batters[1]) if MyModel.battersData.get(batters[1]) is not None else MyModel.averageStrikeRate

                scoresData.loc[j, 'Batter3SR'] = MyModel.battersData.get(batters[2]) if len(
                    batters) > 2 and MyModel.battersData.get(batters[2]) is not None else MyModel.averageStrikeRate
            except Exception as e:
                # print(matchID)
                # print(e)
                # Default Values for things
                scoresData.loc[j, "Batters"] = list()
                scoresData.loc[j, "Bowlers"] = list()
                scoresData.loc[j, 'Batter1SR'] = MyModel.averageStrikeRate
                scoresData.loc[j, 'Batter2SR'] = MyModel.averageStrikeRate
                scoresData.loc[j, 'Batter3SR'] = MyModel.averageStrikeRate

            scoresData.loc[j, "RunsScored"] = ballData["total_run"][(ballData["ID"] == matchID) & (
                ballData["overs"] < 6) & (ballData["innings"] == 2)].sum()
            scoresData.loc[j, "Wickets"] = ballData["isWicketDelivery"][(
                ballData["ID"] == matchID) & (ballData["overs"] < 6) & (ballData["innings"] == 2)].sum()
            j += 1
            mNo += 1
        return scoresData

    def getTeamBatAndBowl(tossWinner, tossDecision, teamsList):
        teams = list()
        team1 = teamsList[0]
        team2 = teamsList[1]
        if tossDecision == "bat":
            teams.append(tossWinner)
            teams.append(team2) if tossWinner == team1 else teams.append(team1)
        else:
            teams.append(
                team2) if tossWinner == team2 else teams.append(teams[0])
            teams.append(tossWinner)
        return teams
    # Getting Batters list from the strings

    def getBatters(batters):
        battersList = list(batters.split(','))
        actualList = list()
        for batter in battersList:
            batter = batter.strip()
            actualList.append(batter)
        return actualList

    venueAvg = {'Narendra Modi Stadiu': 46.07142857142857, 'Eden Gardens, Kolkat': 58.25, 'Wankhede Stadium, Mu': 45.177419354838705, 'Brabourne Stadium, M': 53.147058823529406, 'Dr DY Patil Sports A': 42.32352941176471, 'Maharashtra Cricket ': 46.79545454545455, 'Dubai International ': 45.25, 'Sharjah Cricket Stad': 45.535714285714285, 'Zayed Cricket Stadiu': 50.5, 'Arun Jaitley Stadium': 50.67857142857143, 'MA Chidambaram Stadi': 47.11458333333333, 'Sheikh Zayed Stadium': 43.672413793103445, 'Rajiv Gandhi Interna': 44.06122448979592, 'Dr. Y.S. Rajasekhara': 40.96153846153846, 'Punjab Cricket Assoc': 47.099999999999994, 'Wankhede Stadium': 45.60958904109589, 'M.Chinnaswamy Stadiu': 49.43333333333334, 'Eden Gardens': 46.26623376623377, 'Sawai Mansingh Stadi': 45.04255319148936,
                'Holkar Cricket Stadi': 51.72222222222222, 'M Chinnaswamy Stadiu': 44.65384615384616, 'Feroz Shah Kotla': 46.7, 'Green Park': 55.0, 'Saurashtra Cricket A': 55.3, 'Shaheed Veer Narayan': 38.33333333333333, 'JSCA International S': 38.785714285714285, 'Brabourne Stadium': 51.349999999999994, 'Sardar Patel Stadium': 46.958333333333336, 'Barabati Stadium': 43.57142857142857, 'Subrata Roy Sahara S': 42.4375, 'Himachal Pradesh Cri': 40.55555555555556, 'Nehru Stadium': 39.8, 'Vidarbha Cricket Ass': 44.16666666666667, 'New Wanderers Stadiu': 41.875, 'SuperSport Park': 45.25, 'Kingsmead': 45.93333333333334, 'OUTsurance Oval': 33.5, "St George's Park": 44.785714285714285, 'De Beers Diamond Ova': 40.0, 'Buffalo Park': 39.5, 'Newlands': 40.07142857142857}

    teamAverages = {'Rajasthan Royals': 45.39325842696629, 'Royal Challengers Bangalore': 45.55555555555556, 'Sunrisers Hyderabad': 43.91304347826087, 'Delhi Capitals': 46.58620689655172, 'Chennai Super Kings': 45.3963963963964, 'Gujarat Titans': 42.714285714285715, 'Lucknow Super Giants': 44.875, 'Kolkata Knight Riders': 44.40952380952381, 'Punjab Kings': 47.77777777777778,
                    'Mumbai Indians': 45.214285714285715, 'Kings XI Punjab': 45.71739130434783, 'Delhi Daredevils': 45.63529411764706, 'Rising Pune Supergiant': 50.714285714285715, 'Gujarat Lions': 48.4375, 'Rising Pune Supergiants': 42.142857142857146, 'Pune Warriors': 39.0, 'Deccan Chargers': 42.8974358974359, 'Kochi Tuskers Kerala': 40.857142857142854}

    battersData = dict({'YBK Jaiswal': 148.71794871794873, 'JC Buttler': 154.30839002267572, 'SV Samson': 145.632183908046, 'WP Saha': 138.98531375166888, 'Shubman Gill': 130.89214380825567, 'MS Wade': 119.41747572815532, 'HH Pandya': 122.80701754385966, 'V Kohli': 129.1866028708134, 'F du Plessis': 138.73417721518987, 'RM Patidar': 133.33333333333331, 'Q de Kock': 140.21823850350742, 'KL Rahul': 138.40856924254018, 'M Vohra': 133.51351351351352, 'DJ Hooda': 124.7191011235955, 'PK Garg': 126.3157894736842, 'Abhishek Sharma': 141.12554112554113, 'RA Tripathi': 156.00706713780917, 'JM Bairstow': 153.38809034907598, 'S Dhawan': 132.04066265060243, 'M Shahrukh Khan': 158.33333333333331, 'PP Shaw': 151.46124523506987, 'DA Warner': 145.03642987249546, 'MR Marsh': 180.0, 'RR Pant': 165.78947368421052, 'SN Khan': 167.30769230769232, 'Ishan Kishan': 135.44973544973544, 'RG Sharma': 132.63305322128852, 'D Brevis': 158.33333333333331, 'RD Gaikwad': 122.31404958677685, 'DP Conway': 131.57894736842107, 'MM Ali': 155.19480519480518, 'VR Iyer': 126.26262626262626, 'N Rana': 138.125, 'A Tomar': 50.0, 'SS Iyer': 126.11386138613861, 'Lalit Yadav': 90.0, 'PBB Rajapaksa': 253.125, 'LS Livingstone': 188.88888888888889, 'A Badoni': 28.57142857142857, 'KH Pandya': 259.25925925925924, 'AM Rahane': 123.02667392487751, 'KS Williamson': 117.29166666666666, 'MK Lomror': 117.85714285714286, 'GJ Maxwell': 171.37546468401487, 'RV Uthappa': 138.6564525633471, 'AT Rayudu': 128.13333333333335, 'MS Dhoni': 138.46153846153845, 'S Dube': 171.42857142857142, 'DR Sams': 14.285714285714285, 'Tilak Varma': 170.37037037037038, 'T Stubbs': 0.0, 'HR Shokeen': 20.0, 'R Ashwin': 114.28571428571428, 'KS Bharat': 140.98360655737704, 'KS Sharma': 100.0, 'Ramandeep Singh': 100.0, 'AK Markram': 141.66666666666669, 'B Indrajith': 75.0, 'AJ Finch': 125.54479418886197, 'RK Singh': 242.85714285714283, 'Mandeep Singh': 125.66844919786095, 'B Sai Sudharsan': 137.03703703703704, 'D Padikkal': 127.1880819366853, 'SA Yadav': 162.17228464419475, 'MA Agarwal': 128.57142857142858, 'DJ Mitchell': 80.0, 'MJ Santner': 74.07407407407408, 'MK Pandey': 132.86290322580646, 'Anuj Rawat': 101.14942528735634, 'SS Prabhudessai': 161.53846153846155, 'Shahbaz Ahmed': 126.66666666666666, 'SW Billings': 147.0, 'SP Narine': 172.0, 'JM Sharma': 1400.0, 'V Shankar': 70.83333333333334, 'A Manohar': 108.33333333333333, 'DA Miller': 145.45454545454547, 'P Simran Singh': 104.76190476190477, 'K Gowtham': 133.33333333333331, 'JO Holder': 72.22222222222221, 'E Lewis': 144.96644295302013, 'RA Jadeja': 160.41666666666669, 'TL Seifert': 114.28571428571428, 'Anmolpreet Singh': 116.66666666666667, 'RA Bawa': 100.0, 'SE Rutherford': 41.66666666666667, 'DJ Willey': 75.0, 'N Pooran': 170.27027027027026, 'MP Stoinis': 141.83673469387753,
                       'AR Patel': 286.6666666666667, 'SPD Smith': 149.42263279445726, 'JJ Roy': 122.28571428571429, 'CH Gayle': 144.13830361966504, 'GD Phillips': 100.0, 'SK Raina': 154.79338842975207, 'SS Tiwary': 138.95348837209303, 'AB de Villiers': 151.11111111111111, 'EJG Morgan': 120.10050251256281, 'DT Christian': 62.5, 'DJ Malan': 150.0, 'Virat Singh': 50.0, 'Washington Sundar': 69.23076923076923, 'AD Russell': 329.4117647058823, 'BA Stokes': 142.7027027027027, 'CA Lynn': 158.22784810126583, 'SP Goswami': 102.27272727272727, 'KA Pollard': 151.85185185185185, 'KD Karthik': 140.57971014492753, 'T Banton': 81.81818181818183, 'M Vijay': 120.83333333333333, 'SR Watson': 128.1733746130031, 'SM Curran': 130.18867924528303, 'R Parag': 27.27272727272727, 'R Tewatia': 114.28571428571428, 'N Jagadeesan': 100.0, 'SO Hetmyer': 109.43396226415094, 'KM Jadhav': 128.91566265060243, 'Abdul Samad': 100.0, 'JR Philippe': 107.01754385964912, 'KK Nair': 141.1904761904762, 'C Munro': 151.08695652173913, 'MJ Guptill': 149.04458598726114, 'PA Patel': 126.25766871165645, 'Gurkeerat Singh': 123.07692307692308, 'S Gopal': 75.0, 'Mohammad Nabi': 300.0, 'H Klaasen': 92.85714285714286, 'P Negi': 83.33333333333334, 'Navdeep Saini': 0.0, 'AD Nath': 108.33333333333333, 'JL Denly': 0.0, 'SD Lad': 115.38461538461537, 'NS Naik': 41.17647058823529, 'CA Ingram': 105.55555555555556, 'Shakib Al Hasan': 121.05263157894737, 'MK Tiwary': 125.43103448275863, 'Harbhajan Singh': 135.1851851851852, 'JC Archer': 0.0, 'AD Hales': 145.1219512195122, 'DJM Short': 90.47619047619048, 'BB McCullum': 136.61971830985914, 'JP Duminy': 119.19999999999999, 'YK Pathan': 184.765625, 'G Gambhir': 130.53984575835474, 'RK Bhui': 0.0, 'Yuvraj Singh': 133.50253807106597, 'CH Morris': 271.42857142857144, 'LMP Simmons': 125.62500000000001, 'IR Jaggi': 78.37837837837837, 'Vishnu Vinod': 47.61904761904761, 'SE Marsh': 132.40371845949534, 'DR Smith': 136.09172482552344, 'MC Henriques': 173.61111111111111, 'HM Amla': 136.40776699029126, 'TM Head': 94.11764705882352, 'CJ Anderson': 101.5625, 'MN Samuels': 53.84615384615385, 'PJ Cummins': 100.0, 'SP Jackson': 111.11111111111111, 'STR Binny': 136.0, 'AP Tare': 113.55140186915888, 'K Rabada': 75.0, 'C de Grandhomme': 50.0, 'MJ McClenaghan': 0.0, 'ER Dwivedi': 83.33333333333334, 'Sachin Baby': 0.0, 'NV Ojha': 107.55467196819086, 'UT Khawaja': 139.24050632911394, 'GJ Bailey': 122.90076335877862, 'PP Chawla': 123.07692307692308, 'UBT Chand': 110.06711409395973, 'KP Pietersen': 156.14617940199335, 'MEK Hussey': 115.67635903919088, 'NJ Maddinson': 95.23809523809523, 'V Sehwag': 153.8793103448276, 'GH Vihari': 82.6086956521739, 'MS Bisla': 113.26086956521739, 'RR Rossouw': 91.30434782608695, 'JP Faulkner': 70.0, 'DJ Bravo': 128.44827586206898, 'CM Gautam': 117.1875, 'JA Morkel': 200.0})
