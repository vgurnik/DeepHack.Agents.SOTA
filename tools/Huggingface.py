import requests
from bs4 import BeautifulSoup
import re

class Huggingface:
    
    def __init__(self):
        self.url = "https://huggingface.co"
        self.tasks = None
        
    def get_tasks(self, _type='models'):
        if _type=='models':
            page = requests.get(self.url+'/models')
            soup = BeautifulSoup(page.content, "html.parser")
            self.tasks = json.loads(soup.find(attrs={"data-target": "ModelList"}).attrs["data-props"])['initialValues']['tags']
        else:
            page = requests.get(self.url+'/datasets')
            soup = BeautifulSoup(page.content, "html.parser")
            self.tasks = json.loads(soup.find(attrs={"data-target": "DatasetList"}).attrs["data-props"])['initialValues']['tags']
        self.tasks = {task['label']: task['id'] for task in self.tasks}
        return self.tasks
    
    def get_models(self, task=None, variant='latest', max_amount=5):
        self.get_tasks(_type='models')
        url = self.url+'/models?'
        if not task is None and task in self.tasks.keys():
            url += 'pipeline_tag='+self.tasks[task]+'&'
        if variant=='latest':
            url += 'sort=modified'
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        self.models = json.loads(soup.find(attrs={"data-target": "ModelList"}).attrs["data-props"])['initialValues']['models']
        self.models = [model['id'] for model in self.models[:max_amount]]
        return self.models
    
    def get_datasets(self, task=None, variant='latest', max_amount=5):
        self.get_tasks(_type='datasets')
        url = self.url+'/datasets?'
        if not task is None and task in self.tasks.keys():
            url += 'task_categories=task_categories:'+self.tasks[task]+'&'
        if variant=='latest':
            url += 'sort=modified'
        page = requests.get(url)
        soup = BeautifulSoup(page.content, "html.parser")
        self.datasets = json.loads(soup.find(attrs={"data-target": "DatasetList"}).attrs["data-props"])['initialValues']['datasets']
        self.datasets = [model['id'] for model in self.datasets[:max_amount]]
        return self.datasets
    