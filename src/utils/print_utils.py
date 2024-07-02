"""
This module provides functions and configurations for loading and training machine learning models.

The module includes functions for:
- Printing messages with padding for better readability.

Functions:
- print_with_padding: Print a message with padding for better readability.
"""


def print_with_padding(message):
    """
    Print a message with padding for better readability.
    
    Parameters:
    message (str): The message to be printed.
    
    Example usage:
    --------------
    print_with_padding("This is a test message")
    # Output:
    # ----------
    # This is a test message
    # ----------
    """
    print(f"\n{'-'*10} {message} {'-'*10}\n")
