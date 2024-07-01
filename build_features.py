"""

"""
import re
from urllib.parse import urlparse
import pandas as pd


def count_dot(input_url):
    """
    Count the number of occurrences of '.' in the URL.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    int: The number of occurrences of '.' in the URL.
    """
    return input_url.count(".")


def no_of_dir(input_url):
    """
    Count the number of directories in the URL path.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    int: The number of directories in the URL path.
    """
    urldir = urlparse(input_url).path
    return urldir.count("/")


def no_of_embed(input_url):
    """
    Count the number of embedded URLs in the URL path.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    int: The number of embedded URLs (//) in the URL path.
    """
    urldir = urlparse(input_url).path
    return urldir.count("//")


def shortening_service(input_url):
    """
    Check if the URL is from a known URL shortening service.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    int: 1 if the URL is from a known shortening service, 0 otherwise.
    """
    # List of known URL shortening services patterns
    match = re.search(
        r"bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|"
        r"yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|"
        r"short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|"
        r"doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|"
        r"db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|"
        r"q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|"
        r"j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|"
        r"x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|"
        r"1url\.com|tweez\.me|"
        r"v\.gd|tr\.im|link\.zip\.net",
        input_url,
    )

    # If a match is found, it means the URL is from a known shortening service
    if match:
        return 1  # Return 1 if the URL matches any known shortening service pattern
    return 0  # Return 0 if no match is found


def count_http(input_url):
    """
    Count the number of occurrences of 'http' in the URL.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    int: The number of occurrences of 'http' in the URL.
    """
    return input_url.count("http")


def count_per(input_url):
    """
    Count the number of occurrences of '%' in the URL.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    int: The number of occurrences of '%' in the URL.
    """
    return input_url.count("%")


def count_ques(input_url):
    """
    Count the number of occurrences of '?' in the URL.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    int: The number of occurrences of '?' in the URL.
    """
    return input_url.count("?")


def count_hyphen(input_url):
    """
    Count the number of occurrences of '-' in the URL.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    int: The number of occurrences of '-' in the URL.
    """
    return input_url.count("-")


def count_equal(input_url):
    """
    Count the number of occurrences of '=' in the URL.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    int: The number of occurrences of '=' in the URL.
    """
    return input_url.count("=")


def url_length(input_url):
    """
    Calculate the length of the URL.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    int: The length of the URL.
    """
    return len(str(input_url))


def hostname_length(input_url):
    """
    Calculate the length of the hostname in the URL.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    int: The length of the hostname in the URL.
    """
    return len(urlparse(input_url).netloc)


def suspicious_words(input_url):
    """
    Calculate a suspicious score for a given URL based on the presence
    of specific words or patterns.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    int: The total suspicious score based on the presence of specific
    words or patterns in the URL.
    """
    score_map = {
        "--": 30,
        ".exe": 30,
        ".js": 10,
        ".php": 20,
        "'": 30,
        "admin": 10,
        "administrator": 10,
        "alert": 30,
        "backdoor": 45,
        "bin": 20,
        "carrito": 25,
        "cat": 25,
        "click": 15,
        "cmd": 40,
        "confirm": 20,
        "cookie": 25,
        "cookiesteal": 40,
        "create": 40,
        "credential": 30,
        "DELETE": 50,
        "delay": 35,
        "dir": 30,
        "document": 20,
        "document.": 20,
        "document.cookie": 40,
        "DROP": 50,
        "eval": 30,
        "exec": 30,
        "exploit": 45,
        "expression": 30,
        "fetch": 25,
        "file": 20,
        "FROM": 50,
        "function(": 20,
        "hacker": 35,
        "id": 10,
        "iframe": 25,
        "include": 30,
        "incorrect": 20,
        "inject": 30,
        "INJECTED": 50,
        "javascript": 20,
        "LIKE": 30,
        "location": 30,
        "login": 15,
        "malware": 45,
        "mouseover": 15,
        "onerror": 30,
        "onload": 20,
        "onunload": 20,
        "password": 15,
        "phishing": 45,
        "prompt": 20,
        "proxy": 35,
        "pwd": 15,
        "ransomware": 45,
        "reverse": 30,
        "rootkit": 45,
        "script": 25,
        "SELECT": 50,
        "set": 20,
        "set-cookie": 40,
        "shell": 40,
        "spyware": 45,
        "src=": 25,
        "ssh": 40,
        "steal": 35,
        "TABLE": 50,
        "tamper": 25,
        "tmp": 25,
        "trojan": 45,
        "vaciar": 20,
        "virus": 45,
        "wait": 30,
        "window.": 20,
        "xmlhttprequest": 30,
        "xhr": 20,
    }

    # Create the regex pattern for suspicious words
    pattern = r"(?i)" + r"|".join(re.escape(key) for key in score_map)

    # Find all occurrences of the suspicious words in the URL
    matches = re.findall(pattern, input_url)

    # Calculate the total score based on matches
    total_score = sum(score_map.get(match.lower(), 0) for match in matches)
    return total_score


# import re
# from urllib.parse import urlparse


def digit_count(input_url):
    """
    Count the number of digits in the URL.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    int: The number of digits in the URL.
    """
    digits = 0
    for char in input_url:
        if char.isnumeric():
            digits += 1
    return digits


def letter_count(input_url):
    """
    Count the number of letters in the URL.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    int: The number of letters in the URL.
    """
    letters = 0
    for char in input_url:
        if char.isalpha():
            letters += 1
    return letters


def count_special_characters(input_url):
    """
    Count the number of special characters in the URL.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    int: The number of special characters in the URL.
    """
    special_characters = re.sub(r"[a-zA-Z0-9\s]", "", input_url)
    special_count = len(special_characters)
    return special_count


def number_of_parameters(input_url):
    """
    Count the number of parameters in the URL.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    int: The number of parameters in the URL.
    """
    params = urlparse(input_url).query
    return 0 if params == "" else len(params.split("&"))


def number_of_fragments(input_url):
    """
    Count the number of fragments in the URL.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    int: The number of fragments in the URL.
    """
    frags = urlparse(input_url).fragment
    return 0 if frags == "" else len(frags.split("#")) - 1


# URL is Encoded
def is_encoded(input_url):
    """
    Check if a URL is encoded by looking for the presence of '%' character.

    Parameters:
    input_url (str): The URL to check.

    Returns:
    int: 1 if the URL is encoded, 0 otherwise.
    """
    return int("%" in input_url.lower())


def unusual_character_ratio(input_url):
    """
    Calculate the ratio of unusual characters in a URL.

    Parameters:
    input_url (str): The URL to analyze.

    Returns:
    float: The ratio of unusual characters to the total number of characters in the URL.
    """
    total_characters = len(input_url)
    unusual_characters = re.sub(r"[a-zA-Z0-9\s\-._]", "", input_url)
    unusual_count = len(unusual_characters)
    ratio = unusual_count / total_characters if total_characters > 0 else 0
    return ratio


# Define a function to apply another function to the 'content' column, handling NaNs
def apply_to_content(content, function):
    """
    Apply a function to the content if it is a string, or return 0 if the content is NaN.

    Parameters:
    content: The content to check and apply the function to. It can be any type.
    function: A function to apply to the content if it is a string.

    Returns:
    The result of the function applied to the content if it is a string,
    0 if the content is NaN, or None if the content is neither NaN nor a string.
    """
    if pd.isna(content):
        return 0
    if isinstance(content, str):
        return function(content)
    return None  # ou une autre valeur par défaut appropriée

