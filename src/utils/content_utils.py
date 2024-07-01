"""
Utility functions for content preprocessing.

This module provides various functions to preprocess content for feature extraction.

Functions:
----------
- apply_to_content(content, function): Applies a function to content.
- count_dot(content): Counts the number of dots in the content.
- no_of_dir(content): Counts the number of directories in the content.
- no_of_embed(content): Counts the number of embedded domains in the content.
- count_per(content): Counts the number of percent signs in the content.
- count_ques(content): Counts the number of question marks in the content.
- count_hyphen(content): Counts the number of hyphens in the content.
- count_equal(content): Counts the number of equal signs in the content.
- url_length(content): Calculates the length of the content.
- suspicious_words(content): Checks for suspicious words in the content.
- digit_count(content): Counts the number of digits in the content.
- letter_count(content): Counts the number of letters in the content.
- count_special_characters(content): Counts the number of special characters in the content.
- is_encoded(content): Checks if the content is encoded.
"""

def apply_to_content(content, function):
    return function(content)

def count_dot(content):
    return content.count('.')

def no_of_dir(content):
    return content.count('/')

def no_of_embed(content):
    return content.count('//')

def count_per(content):
    return content.count('%')

def count_ques(content):
    return content.count('?')

def count_hyphen(content):
    return content.count('-')

def count_equal(content):
    return content.count('=')

def url_length(content):
    return len(content)

def suspicious_words(content):
    # Dummy implementation, replace with actual logic
    return int(any(word in content for word in ["suspicious", "malicious"]))

def digit_count(content):
    return sum(c.isdigit() for c in content)

def letter_count(content):
    return sum(c.isalpha() for c in content)

def count_special_characters(content):
    return sum(not c.isalnum() for c in content)

def is_encoded(content):
    return int('%' in content)
