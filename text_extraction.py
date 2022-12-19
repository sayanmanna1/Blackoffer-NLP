import pandas as pd
from bs4 import BeautifulSoup
import requests

df = pd.read_csv('Input.csv')

rows = df.shape[0]

for i in range(rows):
    url = df['URL'][i]
    html_text = requests.get(url, headers = {'User-Agent': 'My User Agent 1.0'}).text
    soup = BeautifulSoup(html_text, 'lxml')
    
    title = soup.title.text
    
    content = soup.find('div', class_='td-post-content')
    with open(f"data/{df['URL_ID'][i]}.txt",'w') as f:
        f.write(f"Title: {title}\n")
        if content != None:
            for para in content.find_all(['p','li','h3','h4','h5']):
                p = para.text
                f.write('\n'+p)