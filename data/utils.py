"""
Data utilities
"""
from bs4 import BeautifulSoup

import os

VALID_TAGS = ["strong", "em", "p", "ul", "li", "br", "code", "pre"]


def sanitize_html_for_web(value, display_code=True):
    """
    Sanitize HTML from all unwanted HTML tags. All <pre><code>...</code></pre>
    sections are replaced with placeholder text is display_code is set to False

    :param value: string with text that needs to be cleaned
    :param display_code: if is True, then the <pre><code>...</code></pre> will not be removed from resulted text
    :return: sanitized text
    """
    soup = BeautifulSoup(value, "html5lib")

    for tag in soup.find_all(True):
        if tag.name not in VALID_TAGS:
            tag.hidden = True
        elif tag.name == "code" and not display_code and tag.parent is not None:
            if tag.parent.name == "pre" and tag.string is not None:  # todo bug?
                tag.string.replace_with("Inserted code --- see details for code expansion")

    return str(soup)


def make_output_dir(output_filename: str, output_dir: str) -> str:
    """
    > This function creates a directory to store some data

    :param output_filename: The name of the file that will be created in the output_dir
    :param output_dir: The directory where the output file will be saved
    """
    if output_filename not in os.listdir(output_dir):
        if output_dir[-1] != "/":
            output_dir += "/"
        os.mkdir(output_dir + output_filename)
    output_dir += "/" + output_filename
    return output_dir









