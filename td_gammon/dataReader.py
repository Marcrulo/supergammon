import csv


                    # Game: episode + 1
                    # Winner: winner
                    # Number of rounds: i
                    # White agent: agents[WHITE].name
                    # White number of victories: wins[WHITE]
                    # White percentage of victory: (wins[WHITE] / tot) * 100
                    # Black agent: agents[BLACK].name
                    # Black number of victories: wins[BLACK]
                    # Black percentage of victory: (wins[BLACK] / tot) * 100
                    # Duration: time.time() - t


def readData(fileName):
    with open(fileName) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        properties = []
        for row in reader:
            properties.append(row)
        
        #print("Game {} was won by {} in {} round(s). Agent {} won {} victories and had a succes rate of {}".format(properties[0], properties[1], properties[2], properties[3], properties[4], properties[5]))
        
        
        
        


readData("stats.csv")

