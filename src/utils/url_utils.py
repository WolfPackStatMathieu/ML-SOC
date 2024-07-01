"""
Utility functions for URL preprocessing.

This module provides various functions to preprocess URLs for feature extraction.

Functions:
----------
- count_dot(url): Counts the number of dots in the URL.
- no_of_dir(url): Counts the number of directories in the URL.
- no_of_embed(url): Counts the number of embedded domains in the URL.
- shortening_service(url): Checks if the URL uses a shortening service.
- count_http(url): Counts the number of "http" in the URL.
- count_per(url): Counts the number of percent signs in the URL.
- count_ques(url): Counts the number of question marks in the URL.
- count_hyphen(url): Counts the number of hyphens in the URL.
- count_equal(url): Counts the number of equal signs in the URL.
- hostname_length(url): Calculates the length of the hostname in the URL.
- suspicious_words(url): Checks for suspicious words in the URL.
- digit_count(url): Counts the number of digits in the URL.
- letter_count(url): Counts the number of letters in the URL.
- url_length(url): Calculates the length of the URL.
- number_of_parameters(url): Counts the number of parameters in the URL.
- number_of_fragments(url): Counts the number of fragments in the URL.
- is_encoded(url): Checks if the URL is encoded.
- count_special_characters(url): Counts the number of special characters in the URL.
- unusual_character_ratio(url): Calculates the ratio of unusual characters in the URL.
"""

def count_dot(url):
    return url.count('.')

def no_of_dir(url):
    return url.count('/')

def no_of_embed(url):
    return url.count('//')

def shortening_service(url):
    # Dummy implementation, replace with actual logic
    return int("short" in url)

def count_http(url):
    return url.lower().count('http')

def count_per(url):
    return url.count('%')

def count_ques(url):
    return url.count('?')

def count_hyphen(url):
    return url.count('-')

def count_equal(url):
    return url.count('=')

def hostname_length(url):
    # Dummy implementation, replace with actual logic
    return len(url.split('/')[2]) if len(url.split('/')) > 2 else 0

def suspicious_words(url):
    # Dummy implementation, replace with actual logic
    return int(any(word in url for word in ["suspicious", "malicious"]))

def digit_count(url):
    return sum(c.isdigit() for c in url)

def letter_count(url):
    return sum(c.isalpha() for c in url)

def url_length(url):
    return len(url)

def number_of_parameters(url):
    return url.count('&')

def number_of_fragments(url):
    return url.count('#')

def is_encoded(url):
    return int('%' in url)

def count_special_characters(url):
    # Dummy implementation, replace with actual logic
    return sum(not c.isalnum() for c in url)

def unusual_character_ratio(url):
    # Dummy implementation, replace with actual logic
    return sum(not c.isalnum() for c in url) / len(url) if len(url) > 0 else 0
