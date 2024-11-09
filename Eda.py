import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
data = read_csv("Book5.csv")
data


data = read_csv("Book5.csv")
position_counts = data['Pos'].value_counts()
colors = ['lightcoral', 'lightskyblue', 'lightgreen', 'lightsalmon', 'lightpink']
plt.figure(figsize=(8, 8))
plt.pie(position_counts, labels=position_counts.index, autopct='%1.1f%%', startangle=140, colors=colors)
plt.title('Distribution of Players by Position')
plt.axis('equal')
plt.show()

data = read_csv("Book5.csv")
filtered_data = data[(data['Age'] >= 15) & (data['Age'] <= 38)]
age_counts = filtered_data['Age'].value_counts().sort_index()
plt.figure(figsize=(10, 6))
plt.bar(age_counts.index, age_counts.values, color='skyblue')
plt.title('Frequency of Ages (15 to 37)')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.xticks(range(15, 38))
plt.show()


data = read_csv("Book5.csv")
team_counts = data['Squad'].value_counts()
plt.figure(figsize=(12, 6))
ax = team_counts.plot(kind='bar', color='skyblue')
plt.xlabel('Team')
plt.ylabel('Number of Players')
plt.title('Number of Players per Team')
plt.xticks(rotation=90)
for i, count in enumerate(team_counts):
     ax.text(i, count, str(count), ha='center', va='bottom')
plt.show()

data = read_csv("Book5.csv")
nation_counts = data['Nation'].value_counts()
plt.figure(figsize=(12, 6))
ax = nation_counts.plot(kind='bar', color='lightblue')
plt.xlabel('Nation')
plt.ylabel('Number of Players')
plt.title('Number of Players per Nation')
plt.xticks(rotation=90)
for i, v in enumerate(nation_counts):
     ax.text(i, v, str(v), va='bottom', ha='center', fontsize=8, color='black')
plt.show()

import seaborn as sns
data = read_csv("Book5.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
nation_goals_scored = data.groupby('Squad')['Goals'].sum().reset_index()
nation_goals_scored = nation_goals_scored.sort_values(by='Goals', ascending=False)
colors_scored = [
    "#00FF00", "#22FF00", "#44FF00", "#44FF00", "#66FF00", "#66FF00", "#66FF00", "#77FF00", "#88FF00", "#99FF00",
    "#CCFF00", "#CCFF00", "#EEFF00", "#FFFF00", "#FFFF00", "#FFFF00", "#FFDD00", "#FF9900", "#FF9900", "#FF7700",
    "#FF1100", "#FF0000", "#FF0000", "#FF0000"
]
cmap_scored = sns.color_palette(colors_scored, n_colors=len(nation_goals_scored))
sns.barplot(x='Goals', y='Squad', data=nation_goals_scored, palette=cmap_scored, orient="h", ax=ax1)
ax1.set_title("Total Goals Scored by Each Team", fontsize=16, fontweight='bold')
ax1.set_xlabel("Total Goals Scored", fontsize=12)
ax1.set_ylabel("Nations", fontsize=12)
goalkeeper_data = data[data['Pos'] == 'GK']
nation_goals_conceded = goalkeeper_data.groupby('Squad')['GoalsAg'].sum().reset_index()
nation_goals_conceded = nation_goals_conceded.sort_values(by='GoalsAg', ascending=True)
colors_conceded = [
    "#00FF00", "#22FF00", "#44FF00", "#44FF00", "#66FF00", "#66FF00", "#66FF00", "#77FF00", "#88FF00", "#99FF00",
    "#CCFF00", "#CCFF00", "#EEFF00", "#FFFF00", "#FFFF00", "#FFFF00", "#FFDD00", "#FF9900", "#FF9900", "#FF7700",
    "#FF1100", "#FF0000", "#FF0000", "#FF0000"
]
cmap_conceded = sns.color_palette(colors_conceded, n_colors=len(nation_goals_conceded))
sns.barplot(x='GoalsAg', y='Squad', data=nation_goals_conceded, palette=cmap_conceded, orient="h", ax=ax2)
ax2.set_title("Total Goals Conceded by Each Team", fontsize=16, fontweight='bold')
ax2.set_xlabel("Total Goals Conceded", fontsize=12)
ax2.set_ylabel("Nations", fontsize=12)
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
top_scoring_nation = nation_goals_scored.nlargest(1, 'Goals')
top_conceding_nation = nation_goals_conceded.nsmallest(1, 'GoalsAg')
for p in ax1.patches:
    width = p.get_width()
    ax1.text(width, p.get_y() + p.get_height() / 2, int(width), ha="left", va="center", fontsize=14, fontweight='bold', color='black')
for p in ax2.patches:
    width = p.get_width()
    ax2.text(width, p.get_y() + p.get_height() / 2, int(width), ha="left", va="center", fontsize=14, fontweight='bold', color='black')
plt.tight_layout(pad=2.0)
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# Load the data
data = pd.read_csv("Book5.csv")
relevant_attributes = [
 'Performance', 'Market Value', '90s', 'Age','Goals','Assists','npxG+xAG',
 'G+A P90','Sh/90', 'SoT%','G/Sh','PrgC/90', 'PrgP/90','PrgR/90',
 'Passes Comp/90','Cmp%','KP/90','SCA90','GCA90','Touches/90','PrgDist/90',
 'Tkl+Int/90','Clr/90',
 'Save%','CS','GoalsAg','AvgDist','GKCmp%','PSxG+/-'
]
# Create a subset of the data with only relevant attributes
subset_data = data[relevant_attributes]
# Compute the correlation matrix
correlation_matrix = subset_data.corr()
# Use hierarchical clustering to arrange attributes
clustered = sns.clustermap(correlation_matrix, cmap='RdYlGn', annot=False, linewidths=.5, figsize=(14, 10))
plt.title('Correlation Analysis')
plt.show()

data = read_csv("Book5.csv")
attributes = ['90s','Goals','Assists','Xg P90','xAG P90','npxG+xAG P90', 'Sh/90', 'KP/90', 'T Att Pen/90']
weights = [3, 5, 3, 5, 3, 5, 1, 1, 1]
forward_data = data[data['Pos'] == 'FW'].copy()
forward_data['Score'] = (forward_data[attributes] * weights).sum(axis=1)
top_forwards = forward_data.sort_values(by='Score', ascending=False).head(5)
plt.figure(figsize=(10, 6))
plt.bar(top_forwards['Player'], top_forwards['Score'], color='skyblue')
plt.title('Top 5 Forwards (FW) Based on Attributes')
plt.xlabel('Player')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.show()


data = read_csv("Book5.csv")
attributes = ['90s','Goals','Assists','npxG+xAG P90','npxG+xAG','PrgC/90','PrgP/90','PrgR/90','Sh/90','KP/90','Through/90','PenAreaPasses/90','T Att 3rd/90']
weights = [3,5,3,5,5,1,1,1,2,2,1,1,1]
forward_data = data[data['Pos'] == 'MF'].copy()
forward_data['Score'] = (forward_data[attributes] * weights).sum(axis=1)
top_forwards = forward_data.sort_values(by='Score', ascending=False).head(5)
plt.figure(figsize=(10, 6))
plt.bar(top_forwards['Player'], top_forwards['Score'], color='skyblue')
plt.title('Top 5 Attacking Midfeilders (AMF) Based on Attributes')
plt.xlabel('Player')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.show()


data = read_csv("Book5.csv")
attributes = ['90s','Red Cards','npxG+xAG P90','PrgP/90','Passes Comp/90','Cmp%','TotDist/90','Live/90','Through/90','Switches/90','Tkl+Int/90','T Mid 3rd/90']
weights = [100,-1,2,2,2,3,2,1,2,2,3,4]
forward_data = data[data['Pos'] == 'MF'].copy()
forward_data['Score'] = (forward_data[attributes] * weights).sum(axis=1)
top_forwards = forward_data.sort_values(by='Score', ascending=False).head(5)
plt.figure(figsize=(10, 6))
plt.bar(top_forwards['Player'], top_forwards['Score'], color='skyblue')
plt.title('Top 5 Central Midfeilders/Central Defensive Midfeilders (CM/CDM) Based on Attributes')
plt.xlabel('Player')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.show()


data = pd.read_csv("Book5.csv")
attributes = ['90s','Red Cards','Passes Comp/90','Cmp%','TotDist/90','Long Cmp%','Tkl+Int/90','Clr/90','Touches/90','T Def Pen/90','T Def 3rd/90','Carries/90']
weights = [100,-1,3,3,3,3,1,3,1,3,3,1]

forward_data = data[data['Pos'] == 'DF'].copy()
forward_data['Score'] = (forward_data[attributes] * weights).sum(axis=1)
top_forwards = forward_data.sort_values(by='Score', ascending=False).head(5)
plt.figure(figsize=(10, 6))
plt.bar(top_forwards['Player'], top_forwards['Score'], color='skyblue')
plt.title('Top 5 Central Backs (CBs) Based on Attributes')
plt.xlabel('Player')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv("Book5.csv")
attributes = ['90s','Save%','CS','GoalsAg','AvgDist','GKCmp%','PSxG+/-']
weights = [1,1,5,-1,1,1,1]

forward_data = data[data['Pos'] == 'GK'].copy()
forward_data['Score'] = (forward_data[attributes] * weights).sum(axis=1)
top_forwards = forward_data.sort_values(by='Score', ascending=False).head(5)
plt.figure(figsize=(10, 6))
plt.bar(top_forwards['Player'], top_forwards['Score'], color='skyblue')
plt.title('Top 5 Goalkeepers (GKs) Based on Attributes')
plt.xlabel('Player')
plt.ylabel('Score')
plt.xticks(rotation=45, ha='right')
plt.show()










