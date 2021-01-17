import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv('research_data.csv')
df = df[df['beginner_win'].notna()]


#plt.scatter(df['lambda'],df['beginner_win'],label="Lambda")
#plt.scatter(df['gamma'],df['beginner_win'],label="Gamma")
#plt.title("Winrate based on gamma and lambda")
#plt.ylabel("Winrate against\nbeginner")

#plt.scatter(df['units'],df['beginner_win'])
#plt.title("Winrate based on amount of hidden units")
#plt.ylabel("Winrate against\nbeginner")

plt.scatter(df['layers'],df['beginner_win'])
plt.title("Winrate based on hidden layers")
plt.ylabel("Winrate against\nbeginner")
plt.xlabel("Hidden layers")



plt.legend()
plt.show()