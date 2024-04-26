import json
import os
import operator

from langgraph.prebuilt import ToolExecutor
from typing import TypedDict, Annotated, Sequence
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langgraph.prebuilt import ToolInvocation
from langchain_core.messages import FunctionMessage
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage, SystemMessage
from langchain.chat_models.gigachat import GigaChat

from tools.Arxiv import Arxiv
from tools.PapersWithCode import PapersWithCode
from tools.Huggingface import Huggingface

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str
hf = Huggingface()
arxiv = Arxiv()
pwc = PapersWithCode()

@tool
def search_papers(query: str) -> dict:
    """Получить список статей, в которых найден запрос query, на английском, с сайта arxiv.org для выбора наиболее релевантных, с ссылками на них.
    Параметр query должен быть списком английских ключевых слов для поиска статей или названием конкретной статьи для её поиска"""
    res = arxiv.search_papers(query)
    return str(res)

@tool
def get_NN_tasks(type: str) -> list:
    """Получить список задач, решаемых нейронными сетями, от huggingface.co для выбора наиболее релевантной.
    Параметр type='models' для поиска задач, для которых есть модели, или type='datasets' для поиска задач, для которых есть датасеты"""
    res = hf.get_tasks(type)
    return list(res.keys())

@tool
def get_NN_models(variant: str='latest', task: str=None) -> list:
    """Получить список моделей с сайта huggingface.co. Параметр variant='latest' для сортировки по последним обновленным/добавленным, или variant='popular' для сортировки по популярности.
    task - один из списка задач, полученных функцией get_NN_tasks, или None (по умолчанию) для поиска из всех моделей"""
    res = hf.get_models(variant, task)
    return list(res)

@tool
def get_NN_datasets(variant: str='latest', task: str=None) -> list:
    """Получить список датасетов с сайта huggingface.co. Параметр variant='latest' для сортировки по последним обновленным/добавленным, или variant='popular' для сортировки по популярности.
    task - один из списка задач, полученных функцией get_NN_tasks, или None (по умолчанию) для поиска из всех датасетов"""
    res = hf.get_models(variant, task)
    return list(res)

@tool
def get_areas() -> list:
    """Получить список обобщенных тем от paperswithcode.com для выбора наиболее релевантной."""
    res = pwc.get_areas()
    return list(res.keys())

@tool
def get_tasks(area: str) -> list:
    """Получить список конкретных задач для выбранной сферы area с сайта paperswithcode.com. Параметр area должен быть строго из списка тем, полученных от вызова функции get_areas."""
    res = pwc.get_tasks(area)
    if res is None:
        return "ERROR: '"+area+"' not one of "+str(list(pwc.get_areas().keys()))
    return list(res.keys())

@tool
def get_task_info(task: str) -> str:
    """Получить информацию о задаче. Параметр task должен быть строго из списка тем, полученных от вызова функции get_tasks.
    Возвращает описание задачи, число статей, наборов данных и бенчмарков по ней."""
    res = pwc.get_task_info(task)
    if res is None:
        return "ERROR: '"+task+"' not one of "+str(list(pwc.get_tasks().keys()))
    description, info = res
    return f"Описание задачи: {description}\nЧисло статей по задаче: {info[0]}, число наборов данных: {info[2]}, число бенчмарков: {info[1]}"

@tool
def get_benchmarks() -> list:
    """Получить список бенчмарков по последней выбранной задаче с сервиса paperswithcode.
    Возвращает список из бенчмарков и наиболее успешно решающих задачу подходов в соответствии с каждым из них.
    Возвращает ошибку, если для задачи нет бенчмарков."""
    results = pwc.get_benchmarks()
    return [str(key) + ': наиболее успешный подход '+str(results[key][1]) for key in results.keys()]

@tool
def get_datasets() -> list:
    """Получить список наборов данных по последней выбранной задаче с сервиса paperswithcode.
    Возвращает ошибку, если для задачи нет датасетов."""
    return list(pwc.get_datasets().keys())

@tool
def get_papers(variant: str) -> list:
    """Получить список статей по последней выбранной задаче с сервиса paperswithcode.
    Параметр variant: 'latest' чтобы получить последние статьи, или 'popular', чтобы получить наиболее цитируемые статьи.
    Возвращает ошибку, если для задачи нет статей."""
    return list(pwc.get_papers(variant).keys())


tools_pwc = [get_benchmarks, get_datasets, get_papers, get_areas, get_tasks, get_task_info]
tools_hf = [get_NN_tasks, get_NN_models, get_NN_datasets]
tools_arxiv = [search_papers]
tools = tools_pwc+tools_hf+tools_arxiv

with open('prompts/system.txt') as f:
    agent_sys_prompt = f.read()
with open('prompts/input.txt') as f:
    agent_input_prompt = f.read()
with open('prompts/controller.txt') as f:
    agent_research_prompt = f.read()
with open('prompts/pwc_agent.txt') as f:
    agent_pwc_prompt = f.read()
with open('prompts/hf_agent.txt') as f:
    agent_hf_prompt = f.read()
with open('prompts/arxiv_agent.txt') as f:
    agent_arxiv_prompt = f.read()
with open('prompts/revisor.txt') as f:
    agent_revisor = f.read()
with open('prompts/revisor_reminder.txt') as f:
    revisor_reminder = f.read()
with open('prompts/greeting.txt') as f:
    greeting = f.read()

if os.path.exists('settings.json'):
    with open('settings.json') as f:
        s = json.load(f)
    creds = s['credentials']
    debug = s['debug']
else:
    creds = input('enter your credentials:')
    debug = True
    with open('settings.json', 'w') as f:
        json.dump({
            'credentials': creds,
            'debug': debug
        }, f, indent=2)

chat = GigaChat(model='GigaChat-Pro-preview', credentials=creds, verify_ssl_certs=False, scope='GIGACHAT_API_CORP',
                profanity_check=False, top_p=0.9, verbose=False, temperature=0)


def revise(model, role, messages, response):
    func = None
    if "function_call" in response.additional_kwargs:
        func = response.additional_kwargs["function_call"]["name"] + '(' + str(
            response.additional_kwargs["function_call"]["arguments"]) + ')'
    revision = chat.invoke([SystemMessage(content=agent_revisor.format(
        role, messages[0].content, [msg.type + ': ' + msg.content for msg in messages[1:]], func)), HumanMessage(
        content=response.content + revisor_reminder.format(
            messages[0].content[messages[0].content.index('Инструкция'):]))])
    while revision.content.startswith('ERROR') or (
            not revision.content.startswith('OK') and not revision.content.startswith('ОК')):
        # print('Ответ '+response.content+' не прошел ревизию:', revision.content)
        if "function_call" in response.additional_kwargs:
            return response
        messages_copy = messages.copy()
        messages_copy.append(response)
        messages_copy.append(HumanMessage(content=revision.content.replace('ERROR:',
                                                                           'SYSTEM: Ты допустил ошибку:') + '\nИсправь её и повтори еще раз. Не извиняйся и не пиши ничего кроме исправленного сообщения.'))
        response = model.invoke(messages_copy, config={"profanity_check": False})
        func = None
        if "function_call" in response.additional_kwargs:
            func = response.additional_kwargs["function_call"]["name"] + '(' + str(
                response.additional_kwargs["function_call"]["arguments"]) + ')'

        revision = chat.invoke([SystemMessage(content=agent_revisor.format(
            role, messages[0].content, [msg.type + ': ' + msg.content for msg in messages[1:]], func)), HumanMessage(
            content=response.content + revisor_reminder.format(
                messages[0].content[messages[0].content.index('Инструкция'):]))], config={"profanity_check": False})
    if revision.content.startswith('OK') or revision.content.startswith('ОК'):
        # print('Ответ прошел ревизию:', revision.content)
        return response


# Задайте функцию, которая определяет нужно продолжать или нет
def should_continue_input(state):
    messages = state['messages']
    last_message = messages[-1]
    # print('INPUT AGENT:', last_message)
    if last_message.content.startswith('ПОЛЬЗОВАТЕЛЬ'):
        return 'user'
    if last_message.content.startswith('РЕЗУЛЬТАТ'):
        print('Система вернула результат работы: ', last_message.content)
        return 'end'
    if last_message.content.startswith('ИССЛЕДОВАТЕЛЬ'):
        return 'research'
    return 'no command'


def should_continue_researcher(state):
    messages = state['messages']
    last_message = messages[-1]
    # print('CONTROL AGENT:', last_message)
    if last_message.content.startswith('PWC'):
        return 'pwc'
    if last_message.content.startswith('HF'):
        return 'hf'
    if last_message.content.startswith('ARXIV'):
        return 'arxiv'
    if last_message.content.startswith('РЕЗУЛЬТАТ'):
        return 'output'
    return 'no command'


def should_continue_browser(state):
    messages = state['messages']
    last_message = messages[-1]
    # print(state["sender"], last_message)
    if "function_call" in last_message.additional_kwargs:
        return 'tool'
    if last_message.content.startswith('РЕЗУЛЬТАТ'):
        return 'output'
    return 'no command'


def should_continue_func(state):
    messages = state['messages']
    last_message = messages[-1]
    func = last_message.name
    if func in [tool.name for tool in tools_pwc]:
        return "pwc"
    if func in [tool.name for tool in tools_hf]:
        return "hf"
    if func in [tool.name for tool in tools_arxiv]:
        return "arxiv"


def prompt_user(state):
    messages = state['messages']
    last_message = messages[-1]
    if last_message.type == "system":
        print(greeting)
    else:
        print("Система отвечает:", last_message.content.replace('ПОЛЬЗОВАТЕЛЬ ', ''))
    inp = input("Пользователь:")
    human_message = HumanMessage(
        content=inp + '\nЕсли в этом запросе содержится сфера деятельности, найди все существующие подходы в этой области.')
    return {"messages": [human_message], "sender": "human"}


# Задайте функцию, которая будет обращаться к модели
def call_model_input(state):
    messages = state['messages']
    if messages[-1].type == 'human':
        input_agent_history.append(messages[-1])
    response = chat.invoke(input_agent_history, config={"profanity_check": False})
    response = revise(chat, 'Входной агент для обработки пользовательского ввода', input_agent_history, response)
    input_agent_history.append(response)
    # Возвращается список, который будет добавлен к существующему списку сообщений
    return {"messages": [response], "sender": "input"}


def call_model_control(state):
    messages = state['messages']
    if state["sender"] == 'input':
        research_agent_history[:] = research_agent_history[:1] + [HumanMessage(content=messages[-1].content)]
    else:
        research_agent_history.append(HumanMessage(content=messages[-1].content))
    response = chat.invoke(research_agent_history, config={"profanity_check": False})
    response = revise(chat, 'Агент для сбора информации на заданную тему с помощью инструментов',
                      research_agent_history, response)
    research_agent_history.append(response)
    return {"messages": [response], "sender": "researcher"}


def call_model_pwc(state):
    messages = state['messages']
    if state["sender"] == 'researcher':
        pwc_agent_history[:] = pwc_agent_history[:1] + [HumanMessage(content=messages[-1].content)]
    elif messages[-1].type == 'function':
        pwc_agent_history.append(messages[-1])
    else:
        pwc_agent_history.append(HumanMessage(
            content="Ты обязан ТОЛЬКО использовать доступные тебе функции: get_benchmarks, get_datasets, get_papers, get_areas, get_tasks, get_task_info, или верни результат сообщением 'РЕЗУЛЬТАТ <твой результат>''"))
    response = chat_pwc.invoke(pwc_agent_history)
    response = revise(chat_pwc, 'Агент для сбора информации на заданную тему с сайта paperswithcode.com',
                      pwc_agent_history, response)
    pwc_agent_history.append(response)
    return {"messages": [response], "sender": "PWC"}


def call_model_hf(state):
    messages = state['messages']
    if state["sender"] == 'researcher':
        hf_agent_history[:] = hf_agent_history[:1] + [HumanMessage(content=messages[-1].content)]
    elif messages[-1].type == 'function':
        hf_agent_history.append(messages[-1])
    else:
        hf_agent_history.append(HumanMessage(
            content="Ты обязан ТОЛЬКО использовать доступные тебе функции: get_NN_tasks, get_NN_models, get_NN_datasets, или верни результат сообщением 'РЕЗУЛЬТАТ <твой результат>''"))
    response = chat_hf.invoke(hf_agent_history)
    response = revise(chat_hf, 'Агент для сбора информации на заданную тему с сайта huggingface.co', hf_agent_history,
                      response)
    hf_agent_history.append(response)
    return {"messages": [response], "sender": "HF"}


def call_model_arxiv(state):
    messages = state['messages']
    if state["sender"] == 'researcher':
        arxiv_agent_history[:] = arxiv_agent_history[:1] + [HumanMessage(content=messages[-1].content)]
    elif messages[-1].type == 'function':
        arxiv_agent_history.append(messages[-1])
    else:
        arxiv_agent_history.append(HumanMessage(
            content="Ты обязан ТОЛЬКО использовать доступные тебе функции: 0, или верни результат сообщением 'РЕЗУЛЬТАТ <твой результат>''"))
    response = chat_arxiv.invoke(arxiv_agent_history)
    response = revise(chat_arxiv, 'Агент для сбора информации на заданную тему с сайта huggingface.co',
                      arxiv_agent_history, response)
    arxiv_agent_history.append(response)
    return {"messages": [response], "sender": "ARXIV"}


# Задайте функцию, которая будет вызывать инструменты
def call_tool(state):
    messages = state['messages']
    # Благодаря условию continue
    # приложение знает, что последнее сообщение содержит вызов функции
    last_message = messages[-1]
    # Создание ToolInvocation из function_call
    action = ToolInvocation(
        tool=last_message.additional_kwargs["function_call"]["name"],
        tool_input=last_message.additional_kwargs["function_call"]["arguments"],
    )
    # Вызов tool_executor и получение ответа
    response = tool_executor.invoke(action)
    # Использование ответа для создания сообщения FunctionMessage
    function_message = FunctionMessage(content=str(response), name=action.tool)
    # Возвращение списка, который будет добавлен к существующему списку сообщений
    return {"messages": [function_message], "sender": "function"}


tool_executor = ToolExecutor(tools)
chat_pwc = chat.bind_tools(tools_pwc)
chat_hf = chat.bind_tools(tools_hf)
chat_arxiv = chat.bind_tools(tools_arxiv)

workflow = StateGraph(AgentState)

workflow.add_node("user", prompt_user)
workflow.add_node("agent_input", call_model_input)
workflow.add_node("agent_controller", call_model_control)
workflow.add_node("agent_pwc", call_model_pwc)
workflow.add_node("agent_hf", call_model_hf)
workflow.add_node("agent_arxiv", call_model_arxiv)
workflow.add_node("action", call_tool)

workflow.set_entry_point("user")

workflow.add_conditional_edges(
    "agent_input",
    should_continue_input,
    {
        "user": "user",
        "research": "agent_controller",
        "no command": "agent_input",
        "end": END
    }
)

workflow.add_conditional_edges(
    "agent_controller",
    should_continue_researcher,
    {
        "pwc": "agent_pwc",
        "hf": "agent_hf",
        "arxiv": "agent_arxiv",
        "output": "agent_input",
        "no command": "agent_controller",
        "end": END
    }
)

workflow.add_conditional_edges(
    "agent_pwc",
    should_continue_browser,
    {
        "tool": "action",
        "output": "agent_controller",
        "no command": "agent_pwc",
        "end": END
    }
)

workflow.add_conditional_edges(
    "agent_hf",
    should_continue_browser,
    {
        "tool": "action",
        "output": "agent_controller",
        "no command": "agent_hf",
        "end": END
    }
)

workflow.add_conditional_edges(
    "agent_arxiv",
    should_continue_browser,
    {
        "tool": "action",
        "output": "agent_controller",
        "no command": "agent_arxiv",
        "end": END
    }
)

workflow.add_conditional_edges(
    "action",
    should_continue_func,
    {
        "pwc": "agent_pwc",
        "hf": "agent_hf",
        "arxiv": "agent_arxiv",
        "end": END
    }
)
workflow.add_edge('user', 'agent_input')
app = workflow.compile()

input_agent_history = [SystemMessage(content=agent_input_prompt)]
research_agent_history = [SystemMessage(content=agent_research_prompt)]
pwc_agent_history = [SystemMessage(content=agent_pwc_prompt)]
hf_agent_history = [SystemMessage(content=agent_hf_prompt)]
arxiv_agent_history = [SystemMessage(content=agent_arxiv_prompt)]
inputs = {"messages": [SystemMessage(content=agent_sys_prompt)]}

for output in app.stream(inputs, config={"recursion_limit": 250}):
    if debug:
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            print(value)
        print("\n---\n")
input('Работа завершена...')