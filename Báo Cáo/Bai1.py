import requests
from bs4 import BeautifulSoup
import pandas as pd

URL = "https://fbref.com/en/comps/9/Premier-League-Stats"

def get_player_data():
    response = requests.get(URL)
    if response.status_code != 200:
        print("Failed to fetch data from FBref.")
        return None
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find("table", {"id": "stats_standard_9"})
    if not table:
        print("Table not found on the page.")
        return None

    headers = [th.text.strip() for th in table.find("thead").find_all("th")]
    rows = []
    for row in table.find("tbody").find_all("tr"):
        if row.find("th", {"scope": "row"}):
            data = [cell.text.strip() if cell.text.strip() else "N/A" for cell in row.find_all("td")]
            rows.append(data)

    df = pd.DataFrame(rows, columns=headers[1:])
    df['Player'] = df['Player'].str.split('\\').str[0]
    df = df[df['Min'].replace("N/A", "0").astype(int) > 90]
    df = df.sort_values(by=['Player', 'Age'], ascending=[True, True])
    df.to_csv("results.csv", index=False)
    return df

data = get_player_data()
if data is not None:
    print("Data has been saved to results.csv.")
