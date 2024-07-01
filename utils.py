
def print_with_padding(text, padding_length=61):
    """
    Print text with padding lines above and below.

    Parameters:
    text (str): The text to print.
    padding_length (int): The length of the padding lines. Default is 61.
    """
    print("-" * padding_length)
    print(f"{'':-^{padding_length}}")
    print(f"{text:-^{padding_length}}")
    print(f"{'':-^{padding_length}}")
    print("-" * padding_length)
