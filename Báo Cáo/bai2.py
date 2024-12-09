import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")
attributes = ['Min', 'Gls', 'Ast', 'xG', 'xA', 'Shots', 'Cmp%', 'Tkl', 'Int', 'PrgC']
top_bottom = {}

for attr in attributes:
    df[attr] = pd.to_numeric(df[attr], errors='coerce')
    top_bottom[attr] = {
        'Top 3': df.nlargest(3, attr)[['Player', attr]],
        'Bottom 3': df.nsmallest(3, attr)[['Player', attr]]
    }

results = []

for attr in attributes:
    median = df[attr].median()
    mean = df[attr].mean()
    std = df[attr].std()
    results.append({'Attribute': attr, 'Type': 'All', 'Median': median, 'Mean': mean, 'Std': std})
    for team, group in df.groupby('Team'):
        median_team = group[attr].median()
        mean_team = group[attr].mean()
        std_team = group[attr].std()
        results.append({'Attribute': attr, 'Type': team, 'Median': median_team, 'Mean': mean_team, 'Std': std_team})

results_df = pd.DataFrame(results)
results_df.to_csv("results2.csv", index=False)

for attr in attributes:
    plt.hist(df[attr].dropna(), bins=20, alpha=0.7, label='All')
    for team, group in df.groupby('Team'):
        plt.hist(group[attr].dropna(), bins=20, alpha=0.5, label=team)
    plt.title(f'Histogram of {attr}')
    plt.xlabel(attr)
    plt.ylabel('Frequency')
    plt.legend(loc='upper right')
    plt.show()

team_scores = df.groupby('Team')[attributes].sum()
best_teams = team_scores.idxmax()

print(best_teams)
