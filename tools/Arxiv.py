import requests
from bs4 import BeautifulSoup
import re

class Arxiv:
    
    def __init__(self):
        self.url = 'https://arxiv.org'
        
    def search_papers(self, term, max_amount=5):
        page = requests.get(self.url+f'/search/?query={term}&searchtype=all&source=header')
        soup = BeautifulSoup(page.content, "html.parser")
        papers = soup.find_all(attrs={"class": "arxiv-result"})
        return {paper.find(attrs={"class": "title"}).text.strip(): paper.find('a').attrs['href'] for paper in papers[:max_amount]}