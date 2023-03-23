from data.documents import Post
from web.search.apps import SearchConfig


def __search_fulltext(search_text, post_start, post_end, request, date_filter):
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
    return search_siamese(result_posts, search_text)


def search_siamese(result_posts: dict, search_text: str) -> dict:
    """
    This function takes in a dictionary of posts and a search text, and returns a dictionary of posts that contain the
    "similarity" tag
    :param result_posts: This is the dictionary of posts that we got from the previous function
    :type result_posts: dict
    :param search_text: The text to search for
    :type search_text: str
    """
    for post in result_posts:
        # Encoding the first post into a sequence of integers.
        encoding_first = SearchConfig.tokenizer.encode(post["title"], max_length=SearchConfig.max_len,
                                                       truncation=True, return_tensors="pt")
        # Encoding the second post into a sequence of integers.
        encoding_second = SearchConfig.tokenizer.encode(search_text,
                                                        max_length=SearchConfig.max_len,
                                                        truncation=True, return_tensors="pt")
        # Using the model to calculate the similarity between the two questions.
        out = SearchConfig.model.forward(encoding_first, encoding_second)
        post["similarity"] = str(out)

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
    # else:
    #     return __search_siamese(search_text, post_start, post_end)
