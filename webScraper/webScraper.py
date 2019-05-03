from bs4 import BeautifulSoup
import pandas as pd
import requests
import re

URL = 'https://www.basketball-reference.com/players/m/mitchdo01/gamelog/2019'
response = requests.get(URL)
used_stats = ['pts','rbs','ast','blk','stl','fgp','fg3','fta','ftm','tov']
soup = BeautifulSoup(response.content, 'html.parser')
points=soup.find_all('td',attrs={'class':'right', 'data-stat':'pts'})
rebounds=soup.find_all('td',attrs={'class':'right', 'data-stat':'trb'})
assists=soup.find_all('td',attrs={'class':'right', 'data-stat':'ast'})
steals=soup.find_all('td',attrs={'class':'right', 'data-stat':'stl'})
blocks=soup.find_all('td',attrs={'class':'right', 'data-stat':'blk'})
field_goal_percentage=soup.find_all('td',attrs={'class':'right', 'data-stat':'fg_pct'})
three_point_percentage=soup.find_all('td',attrs={'class':'right', 'data-stat':'fg3_pct'})
free_throw_attempts=soup.find_all('td',attrs={'class':'right', 'data-stat':'fta'})
free_throws=soup.find_all('td',attrs={'class':'right', 'data-stat':'ft'})
turnovers=soup.find_all('td',attrs={'class':'right', 'data-stat':'tov'})
wins=soup.find_all('tr',attrs={'id':re.compile('')})

#todo probably just needs minutes, and if there is any way to get defense, percentage of team points scored


stats = pd.DataFrame({
    'pts': points,
    'rbs': rebounds,
    'ast': assists,
    'blk': blocks,
    'stl': steals,
    'fgp': field_goal_percentage,
    'fg3': three_point_percentage,
    'ftm': free_throws,
    'fta': free_throw_attempts,
    'tov': turnovers,
    'wns': wins
})
counter=0

#table = soup.find_all('table', class_="row_summable sortable stats_table now_sortable is_sorted")
#divs = soup.find_all('tr', id_="pgl_basic")
def clean_number_stats():
    for i in range(0, len(used_stats)):
        for j in range(0,(len(stats[used_stats[i]]))):
            if i== 6 and j==5:
                print("guck")
            h=stats[used_stats[i]][j].text
            if h=='':
                stats[used_stats[i]][j] = 0.0
            else:
                stats[used_stats[i]][j] = float(h)
            # stats[used_stats[i]][j] = float(stats[used_stats[i]][j].text)

def clean_result_stats():
    for i in range(0, (len(stats['wns']))):
        win_or_loss=stats['wns'][i].find('td',attrs={'class':'center', 'data-stat':'game_result'}).text
        if win_or_loss[0]=='L':
            stats['wns'][i] = 0
        else:
            stats['wns'][i] = 1

def normalize_points():
    max=71
    for i in range(0, len(stats['pts'])):
        stats['pts'][i]=(stats['pts'][i])/max

def normalize_rebounds():
    max=20
    for i in range(0, len(stats['rbs'])):
        stats['rbs'][i]=(stats['rbs'][i])/max

def normalize_assists():
    max=20
    for i in range(0, len(stats['ast'])):
        stats['ast'][i]=(stats['ast'][i])/max

def normalize_blocks():
    max=10
    for i in range(0, len(stats['blk'])):
        stats['blk'][i]=(stats['blk'][i])/max

def normalize_steals():
    max=10
    for i in range(0, len(stats['stl'])):
        stats['stl'][i]=(stats['stl'][i])/max

def normalize_turnovers():
    max=10
    for i in range(0, len(stats['tov'])):
        stats['tov'][i]=(stats['tov'][i])/max


def normalize_freethrows():
    max=20
    for i in range(0, len(stats['ftm'])):
        stats['ftm'][i]=(stats['ftm'][i])/max
    for i in range(0, len(stats['fta'])):
        stats['fta'][i]=(stats['fta'][i])/max


#you can run a model made from lebron, feed it hardens stats, and see how harden would have done if
#he were in lebrons situation

def main():
    clean_number_stats()
    clean_result_stats()
    # normalize_points()
    # normalize_rebounds()
    # normalize_assists()
    # normalize_blocks()
    # normalize_steals()
    # normalize_turnovers()
    # normalize_freethrows()
    print(stats)
    stats.to_csv("/Users/arisemery/CS5665 work/project/donovan2018-19.csv")

    #stats.loc[:, 'pts'] = stats['pts'].str[-5:-1].astype(int)
    # print(type(points))
    # print(len(points))
    # total=0
    # for x in range(0,((len(points)))):
    #     total+=float(points[x].text)
    #     print(points[x].text)
    # print(total/len(points))

if __name__ == "__main__":
    main()