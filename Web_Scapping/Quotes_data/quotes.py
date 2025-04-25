import requests
import csv
from bs4 import BeautifulSoup

URL = "https://www.passiton.com/inspirational-quotes"
r = requests.get(URL)
# print(r.content)

soup = BeautifulSoup(r.content, 'html5lib')

with open('prettify.html', 'w', encoding='utf-8') as file:
    file.write(soup.prettify())
    
print("prettify.html created")

quotes = []

table = soup.find('div', attrs={'id':'all_quotes'})

for i in table.find_all('div', attrs={'class':"col-6 col-lg-3 text-center margin-30px-bottom sm-margin-30px-top"}):
    quote={}
    quote['theme'] = i.h5.text
    quote['url'] = i.a['href']
    quote['img'] = i.img['src']
    quote['lines'] = i.img['alt'].split(" #")[0]
    quote['author'] = i.img['alt'].split(" #")[1]
    quotes.append(quote)
    

filename = 'quotes.csv'
with open(filename, 'w', newline='') as file:
    w = csv.DictWriter(file, ['theme', 'url', 'img', 'lines', 'author'])
    w.writeheader()
    for quote in quotes:
        w.writerow(quote)
