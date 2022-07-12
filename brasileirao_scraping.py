"""original code https://github.com/dataquestio/project-walkthroughs/tree/master/football_matches
scraping 2020 and 2021 season from Brasil football championship
"""
import time
import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = 'https://fbref.com/pt/comps/24/Serie-A-Estatisticas'

years = list(range(2022, 2014, -1))
all_matches = []

for year in years:
    data = requests.get(URL)
    soup = BeautifulSoup(data.text, features='lxml')
    time.sleep(3)
    standings_table = soup.select('table.stats_table')[0]

    links = [l.get("href") for l in standings_table.find_all('a')]
    links = [l for l in links if '/equipes/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]

    previous_season = soup.select("a.prev")[0].get("href")
    URL = f"https://fbref.com{previous_season}"
    team_urls.reverse()
    for team_url in team_urls:
        team_name = team_url.split(
            "/")[-1].replace("-Estatisticas", "").replace("-", " ")
        data = requests.get(team_url)
        time.sleep(3)
        matches = pd.read_html(data.text, match="Resultados e Calendários")[0]
        soup = BeautifulSoup(data.text, features='lxml')
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        data = requests.get(f"https://fbref.com{links[0]}")
        shooting = pd.read_html(data.text, match="Chutes")[0]
        shooting.columns = shooting.columns.droplevel()
        try:
            team_data = matches.merge(
                shooting[["Data", "TC", "CaG", "Dist", "PB", "PT"]], on="Data")
        except ValueError:
            continue
        team_data = team_data[team_data["Camp."] == "Série A"]

        team_data["Temporada"] = year
        team_data["Time"] = team_name
        all_matches.append(team_data)
        time.sleep(10)
        ''' uncomment for safety if you get banned from scraping, you don't lose progress
        kick_out = pd.concat(all_matches)
        kick_out.to_csv(f"{team_name}-{year}.csv")
        '''

print(len(all_matches))
match_df = pd.concat(all_matches)
match_df.columns = [c.lower() for c in match_df.columns]
match_df.to_csv("scraped_matches.csv")
