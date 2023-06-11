from django.apps import AppConfig

from faiss import read_index


class SearchConfig(AppConfig):
    name = 'search'
    indexed_post_bodies = read_index("./search/indexed_data/Body.index")
