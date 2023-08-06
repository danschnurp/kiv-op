# Stackexchange Siamese Posts Search

LaTeX documentation [documentation.pdf](documentation.pdf)

## Create your index with data from [archive.org](https://archive.org/download/stackexchange)

- there are scripts for this too!
- `pip install -r ./data/requirements.txt`
- `data/main_posts_faiss_indexer.py`

`usage: main_posts_faiss_indexer.py [-h] -i INPUT_DIR_PATH [-b BATCH_SIZE] [-s STOP_AT] [--offset OFFSET] [-o OUTPUT_DIR_PATH]`


## How to run:

Look at [releases](https://github.com/danschnurp/kiv-op/releases)

[//]: # (How to run CUSTOM:)

[//]: # ()
[//]: # (- save data for indexing from [Google drive]&#40;https://drive.google.com/drive/folders/19_TiBVPyBIsaVLH1p_Pv6271HpJvGUI7&#41; or [One Drive]&#40;https://onedrive.live.com/?authkey=%21AHBhGMYxGeNPeK0&id=8D9B8AAC1B2B5597%2185158&cid=8D9B8AAC1B2B5597&#41;)

[//]: # (and faiss indexed data from same zip. )

[//]: # (- put **Body.index** to `<project_dir>/web/search/indexed_data/Body.index`)

[//]: # (- put &#40;Badges.xml)

[//]: # (Comments.xml)

[//]: # (PostHistory.xml)

[//]: # (PostLinks.xml)

[//]: # (Posts.xml)

[//]: # (Tags.xml)

[//]: # (Users.xml)

[//]: # (Votes.xml&#41; to `<project_dir>/logstash/data/` directory)

[//]: # ()
[//]: # (- in `docker-compose.yml` set `PAGE: tex` to tor or gamedev or something else )

- run  `docker-compose up`  

- wait until logstash is finished... 

![alt_text](imgs/example.jpg)

- like this picture above

[//]: # (Notes )

[//]: # (- if the app need to be restarted: )

[//]: # (  - delete &#40; **dead_letter_queue plugins queue .lock uuid**&#41; from `<project_dir>/logstash/data/` directory and on docker `/usr/share/logstash/data/` like ont the picture bellow)

[//]: # ()
[//]: # (- do not forget to edit allowed hosts in `web/SiameseSearchWeb/settings.py`)

[//]: # (- if you want api to stack exchange, place the acces token and key to `web/search/apps.py`)

[//]: # (- if css is not loaded, ensure privileges of `web/search/static/`)

[//]: # (- if there are no siemese posts found, ensure privileges of `web/search/indexed_data/Body.index`)

[//]: # (![alt_text]&#40;imgs/delete.jpg&#41;)


### What is done?

- extending [web application](https://github.com/janpasek97/stackoverflow-siamese-network) by:
    - [MQDD - Multimodal Question Duplicity Detection](https://arxiv.org/abs/2203.14093)
which searches for similar questions among the first results of a user query. This approach showed that the inference
of the MQDD model is not fast enough to run 50-500 times in real time on ordinary CPU.

[//]: # (![alt_text]&#40;imgs/Screenshot2.png&#41;)


- creating scripts for file-based index for similarity search [Faiss](https://github.com/facebookresearch/faiss) library and its support in app

[//]: # (![alt_text]&#40;imgs/Screenshot.png&#41;)
- docker support of the web app

and more...

Details you can see in [documentation.pdf](documentation.pdf)
