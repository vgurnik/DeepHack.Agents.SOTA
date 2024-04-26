import requests
from bs4 import BeautifulSoup
import re

class PapersWithCode:
    
    def __init__(self):
        self.url = "https://paperswithcode.com"
        page = requests.get(self.url+'/sota')
        soup = BeautifulSoup(page.content, "html.parser")
        self.areas = {area.find('a').text: area.find('a').attrs['href'] for area in soup.find_all(attrs={'class':'row task-group-title'})}
        self.current_area = None
        self.cards = None
        self.current_task = None
        self.current_task_soup = None
        self.papers = None
        self.benchmarks = None
        self.datasets = None
        
    def get_areas(self):
        return self.areas
    
    def get_tasks(self, area=None):
        if area is None:
            area = self.area
        if not area in self.areas.keys():
            return None
        page = requests.get(self.url+self.areas[area])
        soup = BeautifulSoup(page.content, "html.parser")
        cards = {card.find('h1').text: card.find('a').attrs['href'] for card in soup.find_all(attrs={'class':'card'})}
        self.cards = cards
        return cards
    
    def get_task_info(self, task, cards=None):
        if cards is None:
            cards = self.cards
        if (cards is None) or (not task in cards.keys()):
            return None
        page = requests.get(self.url+cards[task])
        soup = BeautifulSoup(page.content, "html.parser")
        self.current_task_soup = soup
        description = soup.find(attrs={'class':'description'}).text.strip()
        info = soup.find(attrs={'class':'artefact-information'}).find('p').text
        data = list(map(int, re.findall('[0-9]+', info)))
        self.papers, self.benchmarks, self.datasets = data
        return description, data
        
    def print_info(self, task, descr, info):
        print(descr)
        print(task, r'has {} papers, {} benchmarks, {} datasets'.format(*info))
        
    def get_benchmarks(self):
        if self.benchmarks is None or self.benchmarks == 0:
            return None
        table = self.current_task_soup.find(attrs={"id": "benchmarksTable"}).find('tbody').find_all('tr')
        table = [tr.find_all('td') for tr in table]
        return {td[1].text.strip(): (td[1].find('a').attrs['href'], td[2].text.strip(), td[2].find('a').attrs['href']) for td in table}
    
    def get_datasets(self):
        if self.datasets is None or self.datasets == 0:
            return None
        table = self.current_task_soup.find(attrs={"class": "task-datasets"}).find_all('li')
        return {li.text.strip(): li.find('a').attrs['href'] for li in table}
    
    def get_papers(self, variant='latest'):
        if self.papers is None or self.papers == 0:
            return None
        links = pwc.current_task_soup.find_all('a', attrs={'class': 'list-papers-button'})
        if variant == 'latest':
            link = links[2].attrs['data-call-url']
        elif variant == 'popular':
            link = links[0].attrs['data-call-url']
        page = requests.get(self.url + link)
        soup = BeautifulSoup(page.content, "html.parser")
        papers = soup.find_all('h1')
        return {paper.text.strip(): paper.find('a').attrs['href'] for paper in papers}
    