"""These contains functions for cleaning data in pandas."""

import logging

from unidecode import unidecode

logger = logging.getLogger(__name__)


def _replace_elements(
    somestring,
    elem_list=(
        ["á", "a"],
        ["é", "e"],
        ["í", "i"],
        ["ó", "o"],
        ["ú", "u"],
        ["ý", "y"],
        ["à", "a"],
        ["è", "e"],
        ["ì", "i"],
        ["ò", "o"],
        ["ù", "u"],
        ["ä", "a"],
        ["ë", "e"],
        ["ï", "i"],
        ["ö", "o"],
        ["ü", "u"],
        ["ÿ", "y"],
        ["â", "a"],
        ["ê", "e"],
        ["î", "i"],
        ["ô", "o"],
        ["û", "u"],
        ["ã", "a"],
        ["õ", "o"],
        ["@", "a"],
        ["ñ", "n"],
    ),
) -> str:
    """Replace elements in a string."""
    for elems in elem_list:
        somestring = str(somestring).replace(elems[0], elems[1])
    return somestring


def _unidecode_strings(
    somestring: str,
    characters_to_replace=(
        "(",
        ")",
        "*",
        " ",
        ":",
        ".",
        "-",
        ";",
        "<",
        "?",
        "/",
        ",",
        "'",
        "____",
        "___",
        "__",
        "'",
        "&",
    ),
) -> str:
    """Unidecode string.

    It takes a string, converts it to unicode, then converts it to ascii,
    then lowercases it, then replaces all the characters in the list
    with underscores.

    Args:
      somestring (str): The string you want to unidecode.
      characters_to_replace: a list of characters to replace with an underscore

    Returns:
      A string formatted.
    """
    somestring = somestring.lower()
    u = unidecode(somestring, "utf-8")
    formated_string = unidecode(u)
    for character in characters_to_replace:
        formated_string = formated_string.replace(character, "_")
    if "_" in formated_string:
        last_underscore_index = formated_string.rindex("_")
        if last_underscore_index == len(formated_string) - 1:
            formated_string = formated_string[:-1]
    formated_string = _replace_elements(formated_string)
    return formated_string
