set TOKENIZERS_PARALLELISM=true && \
 source %PROJECT_DIR%/venv/bin/activate && \
 python  %PROJECT_DIR%/data/main_posts_faiss_indexer.py -i %INTPUT_DIR% -b %BATCH_MAXIMUM_SIZE%

