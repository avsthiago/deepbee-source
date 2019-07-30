from bs4 import BeautifulSoup
import re 

urls_file = str()

with open("IPBcloud.htm", "r") as file:
    lines = file.readlines()
    lines = [i.strip() for i in lines]
    urls_file = "\n".join(lines)

soup = BeautifulSoup(urls_file)


with open("urls_with_name.txt", "w") as file:
    for link in soup.findAll('tr'):
        if "Last Update" not in link.text:
            name = link.findAll('td')[1].text
            url = link.findAll('td')[4].find('a').get("href")
            file.write(f"{name},{url}\n")
