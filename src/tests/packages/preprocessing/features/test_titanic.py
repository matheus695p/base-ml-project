import math
import numpy as np
from project.packages.preprocessing.features.titanic import (
    _parse_passenger_ticket,
    _parse_passenger_cabin,
    _extract_cabin_number,
)


class TestParsePassengerTicket:
    def test_parse_passenger_ticket_with_valid_input(self):
        # Arrange
        passenger_ticket = "PC 12345"
        expected_output = ("PC", 12345)
        # Act
        result = _parse_passenger_ticket(passenger_ticket)

        # # Assert
        assert result == expected_output

    def test_parse_passenger_ticket_with_invalid_input(self):
        # Arrange
        passenger_ticket = "Invalid Ticket"
        expected_output = ("Invalid", np.nan)

        # Act
        result = _parse_passenger_ticket(passenger_ticket)

        # Assert
        assert result == expected_output


class TestParsePassengerCabin:
    def test_parse_passenger_cabin_with_valid_input(self):
        # Arrange
        passenger_cabin = "C123"
        expected_output = "C"

        # Act
        result = _parse_passenger_cabin(passenger_cabin)

        # Assert
        assert result == expected_output

    def test_parse_passenger_cabin_with_unknown_input(self):
        # Arrange
        passenger_cabin = "unknown"
        expected_output = "unknown"

        # Act
        result = _parse_passenger_cabin(passenger_cabin)

        # Assert
        assert result == expected_output


class TestExtractCabinNumber:
    def test_extract_cabin_number_with_valid_input(self):
        # Arrange
        cabin_string = "C123"
        expected_output = 123

        # Act
        result = _extract_cabin_number(cabin_string)

        # Assert
        assert result == expected_output

    def test_extract_cabin_number_with_invalid_input(self):
        # Arrange
        cabin_string = "Invalid Cabin"

        # Assert
        assert math.isnan(_extract_cabin_number(cabin_string))
