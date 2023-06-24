from django.apps import AppConfig

from faiss import read_index


class SearchConfig(AppConfig):
    name = 'search'
    indexed_post_bodies = read_index("./search/indexed_data/Body.index")
    # todo replace with real key and access token
    STACK_EXCHANGE_KEY = 0
    STACK_EXCHANGE_ACCESS_TOKEN = 0
