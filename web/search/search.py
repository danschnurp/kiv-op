import numpy as np
from torch import reshape

from .documents import Post, PostLink
from .apps import SearchConfig


def __encode_question(question, tokenizer, model, max_length=512, this_device="cpu"):
    """
    This function encodes a given question using a tokenizer and a pre-trained language model, with a specified maximum
    length.

    :param question: The input question that needs to be encoded
    :param tokenizer: The tokenizer is a tool used to convert text into numerical values that can be processed by model.
    :param model: The model parameter refers to the pre-trained language model.
    :param max_length: The maximum length of the input sequence that the model can handle. If the input sequence is longer
    than this, it will be truncated, defaults to 512 (optional)
    """
    encoded_question = tokenizer(
        question, max_length=max_length,
        padding="max_length",
        return_token_type_ids=True,
        truncation=True, return_tensors="pt")
    encoded_question.to(this_device)

    # reshaping (adding dimension) to have a batch size
    encoded_question.data["input_ids"] = reshape(
        encoded_question.data["input_ids"],
        (1, encoded_question.data["input_ids"].shape[1]))
    encoded_question.data["token_type_ids"] = reshape(
        encoded_question.data["token_type_ids"],
        (1, encoded_question.data[
            "token_type_ids"].shape[1]))
    encoded_question.data["attention_mask"] = reshape(
        encoded_question.data["attention_mask"],
        (1, encoded_question.data[
            "attention_mask"].shape[1]))
    # calling the pre-trained language model UWB-AIR/MQDD-duplicates
    _, encoded_question_result = model(**encoded_question)

    return np.squeeze(encoded_question_result.detach().cpu().numpy().tolist())


def run_faiss(vectorized_post, normalized_post, max_results=50):
    normalized_post[0, :len(vectorized_post)] = vectorized_post

    _, result_ids, _ = SearchConfig.indexed_post_bodies.search_and_reconstruct(normalized_post, max_results)
    return np.squeeze(result_ids)


def __search_fulltext(search_text, search_code, post_start, post_end, request, date_filter,
                      page=SearchConfig.INDEXED_SITE, siamese_search=False):
    """
    Perform fulltext search in order to find posts that match with given search_text string

    :param search_text: text from user for which all matched are gonna be searched for
    :param post_start: post search start offset - pagination parameter
    :param post_end: post search end offset - pagination parameter
    :param request: request object which carries important GET and POST parameters
    :param date_filter: date filter settings for search
    :return: display information of all found articles
    """

    if siamese_search:
        try:
            question = f"<CLS>{search_text}<SEP>{search_code}<SEP>"
            vectorized_post = __encode_question(question, SearchConfig.tokenizer, SearchConfig.model, )
            normalized_post = np.zeros((1, SearchConfig.indexed_post_bodies.d))
            post_ids = list(run_faiss(normalized_post=normalized_post, vectorized_post=vectorized_post))
            # gets siamese posts from vectored posts
            posts_search = []

            for post_id in post_ids[:10]:
                try:
                    post = Post.search().filter("term", post_ID=post_id).filter("match", page=page).execute()
                    result_posts = Post.get_display_info_for_posts(post.hits)
                    posts_search.extend(result_posts)
                except Exception:
                    continue
        except Exception:
            return []
        return posts_search

    else:
        posts_search = Post.search().filter("term", post_type=1).query("multi_match", query=search_text,
                                                                       fields=["title", "text"])[post_start:post_end]

        # determine filters
        if request.GET.getlist("pages") and request.GET.get("pages", "all") != "all":  # pages filter
            pages = request.GET.getlist("pages")
            posts_search = posts_search.filter("terms", page=pages)

        if request.GET.get("with_answer", None) is not None:  # with answer only filter
            posts_search = posts_search.filter("exists", field="accepted_answer_ID")

        # date filter
        posts_search = posts_search.filter("range",
                                           creation_date={"gte": date_filter["start"], "lt": date_filter["end"]})
        # date filter always has at least a default value

        posts_response = posts_search.execute()
        result_posts = Post.get_display_info_for_posts(posts_response.hits)

        post_links_search = [PostLink.search().query(
            "match", post_ID=i["post_ID"],

        ) for i in result_posts]

        links_response = [i.execute() for i in post_links_search]
        result_links = [PostLink.get_display_info_for_links(i.hits, result_posts[0]["page"]) for i in links_response]

        for k in result_links:
            if k:
                for i in k:
                    for j in range(len(result_posts)):
                        if result_posts[j]["post_ID"] == i["post_ID"]:
                            if "linked_posts" not in result_posts[j]:
                                result_posts[j]["linked_posts"] = i['related_question_id']
                                result_posts[j]["linked_posts_titles"] = i['related_question_title']
                            # else:
                            #     result_posts[j]["linked_posts"].append(i['related_question_id'])
                            #     result_posts[j]["linked_posts_titles"].append(i['related_question_title'])
        # todo issue with multiple parsing linked posts with for pagination
        return result_posts


def search_siamese_faissly(post_id, max_results=5):
    """
    This function searches for similar posts to a given post using the Siamese algorithm and Faiss library, with a maximum
    number of results specified.
    
    :param post_id: The ID of the post that we want to find similar posts for
    :param max_results: The maximum number of search results to return, defaults to 5 (optional)
    """
    normalized_post = np.zeros((1, SearchConfig.indexed_post_bodies.d))
    try:
        vectorized_post = SearchConfig.indexed_post_bodies.reconstruct(post_id)
        result_ids = run_faiss(vectorized_post, normalized_post, max_results=5)
        return result_ids[result_ids != post_id]
    except Exception:
        return []


def get_post_links_from_users(result_post, page):
    post_links_search = PostLink.search().query("match", post_ID=result_post)
    links_response = post_links_search.execute()
    return PostLink.get_display_info_for_links(links_response.hits, page) if len(links_response.hits) > 0 else None


def search(search_type, search_text, search_code, page, posts_per_page, request, date_filter):
    """
    Perform selected type of post search and return display information of all resulted posts

    :param search_type: type of search that is gonna be performed - values: {fulltext, siamese}
    :param search_text: text from user for which all matched are gonna be searched for
    :param search_code: code from user for which all matched are gonna be searched for
    :param page: page from which the articles are gonna be searched for
    :param posts_per_page: how many posts shall be found
    :param request: request object which carries important GET and POST parameters
    :param date_filter: date filter settings for search
    :return: display information of all found articles
    """
    if search_text == "":
        return []
    post_start = (page - 1) * posts_per_page
    post_end = (page * posts_per_page) + 1
    if search_type == "fulltext":
        return __search_fulltext(search_text, "", post_start, post_end, request, date_filter)
    elif search_type == "siamese":
        return __search_fulltext(search_text, search_code, post_start, post_end, request, date_filter,
                                 siamese_search=True)
    else:
        return __search_fulltext(search_text, "", post_start, post_end, request, date_filter)
