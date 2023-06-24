import requests

from .apps import SearchConfig


def post_question(title, body, tags, site):
    if SearchConfig.STACK_EXCHANGE_KEY == 0 or SearchConfig.STACK_EXCHANGE_ACCESS_TOKEN == 0:
        return
    url = "https://api.stackexchange.com/2.3/questions/add"
    params = {
        "title": title,
        "body": body,
        "tags": tags,
        "site": site,
        "key": SearchConfig.STACK_EXCHANGE_KEY,
        "access_token": SearchConfig.STACK_EXCHANGE_ACCESS_TOKEN
    }
    response = requests.post(url, params=params)
    return response.json()