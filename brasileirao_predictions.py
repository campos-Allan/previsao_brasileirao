""" predicting games on BR football championship
inspired by
https://github.com/dataquestio/project-walkthroughs/blob/master/football_matches/prediction.ipynb
"""

from datetime import date
from typing import List
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.ensemble import RandomForestClassifier
import pandas as pd


class MissingDict(dict):
    """some club names are different in home and away data,
    this is to map the names and make everything the same

    Args:
        dict (_type_): old names and new names
    """

    def __missing__(self, key):
        return key


def rolling_averages(group: pd.DataFrame, cols: List, new_cols: List) -> pd.DataFrame:
    """get average stats from 3 last games

    Args:
        group (pd.DataFrame): team DataFrame
        cols (List): old stats
        new_cols (List): new average stats from last 3 games

    Returns:
        pd.DataFrame: team data with rolling averages
    """
    group = group.sort_values("data")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group


def make_predictions(data: pd.DataFrame, predictors: List) -> pd.DataFrame:
    """fitting and predicting

    Args:
        data (pd.DataFrame): match data
        predictors (List): x attributes

    Returns:
        pd.DataFrame: df comparison between test and predicted
    """
    train = data[data["data"] < date(2021, 8, 1)]
    test = data[data["data"] > date(2021, 8, 1)]
    rf.fit(train[predictors], train["3pts"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(
        dict(actual=test["3pts"], predicted=preds), index=test.index)
    precision = precision_score(test["3pts"], preds)
    acc = accuracy_score(test['3pts'], preds)
    print(f'Precision:{precision}')
    print(f'Accuracy:{acc}')
    return combined


# PREPROCESSING ~ fixing problems
df_matches = pd.read_csv('scraped_matches.csv', index_col=0)
df_matches.columns = [new_name.lower() for new_name in df_matches.columns]
df_matches.drop(['relatório da partida', 'dist', 'notas', 'camp.'],
                axis=1, inplace=True)  # empty columns
print(df_matches.shape)
# scraped_matches2.csv has data from 2015 onwards
# some duplicated games, bc 2020 championship entered 2021 year
df_matches['data'] = pd.to_datetime(
    df_matches['data'], format='%Y-%m-%d').dt.date

df_matches_new = df_matches[df_matches['temporada'] != 2021]
df_matches_2021 = df_matches[df_matches['temporada'] == 2021]

df_matches_2021 = df_matches_2021[df_matches_2021['data'] > date(
    2021, 4, 30)]  # taking off 2020 games duplicated as 2021 games
df_matches = pd.concat([df_matches_new, df_matches_2021])

print(df_matches.shape)  # 38 matches * 20 teams * 2 seasons = 1520 games
# 2016 has two games less in scraped_matches2.csv bc match cancellation

# público (attendance) columns has some irreal values
df_matches['público'] = df_matches['público'].apply(
    lambda x: 16 if x > 65 else x)

# PREPROCESSING ~ formatting
df_matches['3pts'] = (df_matches['resultado'] == 'V').astype('int')  # target
df_matches['casa'] = df_matches['local'].astype(
    'category').cat.codes  # home or away
df_matches['adversário_código'] = df_matches['oponente'].astype(
    'category').cat.codes  # opponent code
df_matches['hora'] = df_matches['horário'].str.replace(
    ':.+', '', regex=True).astype('int')  # hour
df_matches['dia_da_semana'] = df_matches['data'].map(
    lambda x: x.weekday())  # day of the week

cols = ["gp", "gc", "tc", "cag", "pb", "pt"]
# goals pro, goals against, total shoots, shoots on goal, penalty goals, penaltys taken
new_cols = [f"{c}_rolling" for c in cols]

matches_rolling = df_matches.groupby("time").apply(
    lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('time')

matches_rolling.index = range(matches_rolling.shape[0])

# MODELLING
rf = RandomForestClassifier(
    n_estimators=50, min_samples_split=10, random_state=9)
predictors = ["casa", "adversário_código",
              "hora", "dia_da_semana"]

combined = make_predictions(
    matches_rolling, predictors + new_cols)

combined = combined.merge(matches_rolling[[
    "data", "time", "oponente", "resultado"]], left_index=True, right_index=True)

map_values = {"Atletico Mineiro": "Atlético Mineiro", "Ceara": "Ceará",
              "Atletico Goianiense": "Atl Goianiense", "Atletico Paranaense": "Atl Paranaense",
              "Gremio": "Grêmio", "Sao Paulo": "São Paulo", "Vasco da Gama": "Vasco",
              "Goias": "Goiás", "Botafogo RJ": "Botafogo (RJ)", "Cuiaba": "Cuiabá",
              "America MG": "América (MG)"}
mapping = MissingDict(**map_values)
combined["novo_time"] = combined["time"].map(mapping)

merged = combined.merge(
    combined, left_on=["data", "novo_time"], right_on=["data", "oponente"])

new_precision = merged[(merged["predicted_x"] == 1) & (
    merged["predicted_y"] == 0)]["actual_x"].value_counts()
# discarding predictions where the model predicted both teams to win or lose
print(f'New Precision: {new_precision[1]/(new_precision[0]+new_precision[1])}')

'''
FUTURE UPDATES:
-> test games from 2022, instead of half 2021 season
-> using other attributes
-> using other algorithms
-> create spi for each team based on last year performance and transfermarket value
-> base new predictions around that
'''
