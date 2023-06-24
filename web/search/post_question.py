import requests

from web.SiameseSearchWeb.settings import STACK_EXCHANGE_KEY, STACK_EXCHANGE_ACCESS_TOKEN


def post_question(title, body, tags, site):
    if STACK_EXCHANGE_KEY == 0 or STACK_EXCHANGE_ACCESS_TOKEN == 0:
        return
    url = "https://api.stackexchange.com/2.3/questions/add"
    params = {
        "title": title,
        "body": body,
        "tags": tags,
        "site": site,
        "key": STACK_EXCHANGE_KEY,
        "access_token": STACK_EXCHANGE_ACCESS_TOKEN
    }
    response = requests.post(url, params=params)
    return response.json()