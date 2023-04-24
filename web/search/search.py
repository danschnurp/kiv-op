import time
from itertools import permutations

from data.documents import Post, PostLink
from web.search.apps import SearchConfig
from web.search.brain.question_encoder import encode_question


def __search_fulltext(search_text, post_start, post_end, request, date_filter, siamese_search=False):
    """
    Perform fulltext search in order to find posts that match with given search_text string

    :param search_text: text from user for which all matched are gonna be searched for
    :param post_start: post search start offset - pagination parameter
    :param post_end: post search end offset - pagination parameter
    :param request: request object which carries important GET and POST parameters
    :param date_filter: date filter settings for search
    :return: display information of all found articles
    """
    posts_search = Post.search().filter("term", post_type=1).query("multi_match", query=search_text,
                                                                   fields=["title", "text"])[post_start:post_end]
    # post_links = PostLink.search().filter("term", post_type=1).query(
    # #     todo
    # )[post_start:post_end]

    # determine filters
    if request.GET.getlist("pages") and request.GET.get("pages", "all") != "all":  # pages filter
        pages = request.GET.getlist("pages")
        posts_search = posts_search.filter("terms", page=pages)

    if request.GET.get("with_answer", None) is not None:  # with answer only filter
        posts_search = posts_search.filter("exists", field="accepted_answer_ID")

    # date filter
    posts_search = posts_search.filter("range", creation_date={"gte": date_filter["start"], "lt": date_filter["end"]})
    # date filter always has at least a default value

    posts_response = posts_search.execute()
    result_posts = Post.get_display_info_for_posts(posts_response.hits)
    if siamese_search:
        return search_siamese(result_posts)
    else:
        return result_posts


def search_siamese(result_posts: dict) -> dict:
    """
    This function takes in a dictionary of posts and a search text, and returns a dictionary of posts that contain the
    "similarity" tag
    :param result_posts: This is the dictionary of posts that we got from the previous function
    :type result_posts: dict
    :param search_text: The text to search for
    :type search_text: str
    """
    encoded_posts = [
        encode_question(post["text"], SearchConfig.tokenizer) for post in result_posts]
    t1 = time.time()
    combinations = {}
    # creates variation pairs for (first 3 and all) because it is still slow
    for index, i in enumerate(encoded_posts[:3]):
        for jndex, j in enumerate(encoded_posts):
            if i != j:
                combinations[str(index) + "_" + str(jndex)] = [i, j]
    # searching for related questions among the results
    result_combinations = [SearchConfig.model.forward(dict(out[0].data, **out[1].data))
                           for out in combinations.values()]
    # assign duplicate occurrences
    for combination, result in zip(combinations.keys(), result_combinations):
        if "DUPLICATE" == result:
            post = result_posts[int(combination.split("_")[0])]
            if "related_question_ids" in post and "related_question_titles" in post:
                post["related_question_ids"].append(result_posts[int(combination.split("_")[1])]["post_ID"])
                post["related_question_titles"].append(result_posts[int(combination.split("_")[1])]["title"])
            else:
                post["related_question_ids"] = [result_posts[int(combination.split("_")[1])]["post_ID"]]
                post["related_question_titles"] = [result_posts[int(combination.split("_")[1])]["title"]]

    # zip for better structure parsing to template
    for post in result_posts:
        if "related_question_ids" in post and "related_question_titles" in post:
            post["related_questions"] = zip(post["related_question_ids"], post["related_question_titles"])
    print(time.time() - t1)

    return result_posts


def search(search_type, search_text, page, posts_per_page, request, date_filter):
    """
    Perform selected type of post search and return display information of all resulted posts

    :param search_type: type of search that is gonna be performed - values: {fulltext, siamese}
    :param search_text: text from user for which all matched are gonna be searched for
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
        return __search_fulltext(search_text, post_start, post_end, request, date_filter)
    else:
        return __search_fulltext(search_text, post_start, post_end, request, date_filter, siamese_search=True)
