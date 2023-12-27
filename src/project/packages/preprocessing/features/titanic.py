import numpy as np


def _parse_passenger_ticket(passenger_ticket: str) -> str:
    """
    Parse the passenger ticket information and extract the ticket base and number.

    Args:
        passenger_ticket (str): The passenger's ticket information.

    Returns:
        tuple: A tuple containing the ticket base (str) and ticket number (int).
               If the ticket number cannot be extracted, it will be set to NaN.

    Example:
        >>> _parse_passenger_ticket("PC 12345")
        ('PC', 12345)
    """
    if " " in passenger_ticket:
        parsed_ticket = passenger_ticket.split(" ")
    else:
        parsed_ticket = ["unknown", passenger_ticket]

    try:
        ticket_number = int("".join(filter(str.isdigit, parsed_ticket[1])))
    except Exception:
        ticket_number = np.nan

    return parsed_ticket[0], ticket_number


def _parse_passenger_cabin(passenger_cabin: str) -> str:
    """
    Extract the passenger cabin level from the given passenger cabin information.

    Args:
        passenger_cabin (str): The passenger's cabin information.

    Returns:
        str: The extracted passenger cabin level (the first character of the cabin information).
             If the cabin information is "unknown," it will be returned as "unknown."

    Example:
        >>> _passenger_cabin("C123")
        'C'
    """
    passenger_cabin = passenger_cabin.strip()
    if passenger_cabin == "unknown":
        passenger_cabin_level = "unknown"
    else:
        passenger_cabin_level = passenger_cabin[0]
    return passenger_cabin_level


def _extract_cabin_number(cabin_string: str) -> str:
    """Extract cabin number from cabin string."""
    try:
        cabin_number = int("".join(filter(str.isdigit, cabin_string)))
    except Exception:
        cabin_number = np.nan
    return cabin_number
