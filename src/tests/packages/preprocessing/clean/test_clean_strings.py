from project.packages.preprocessing.clean.clean_strings import _replace_elements, _unidecode_strings


class TestReplaceElements:
    def test_replace_elements_with_default_list(self):
        # Arrange
        input_string = "áéíóúýàèìòùäëïöüÿâêîôûãõ@ñ"
        expected_output = "aeiouyaeiouaeiouyaeiouaoan"

        # Act
        result = _replace_elements(input_string)

        # Assert
        assert result == expected_output

    def test_replace_elements_with_custom_list(self):
        # Arrange
        input_string = "áéíóúýàèìòùäëïöüÿâêîôûãõ@ñ"
        custom_element_list = [("á", "a"), ("é", "e")]
        expected_output = "aeíóúýàèìòùäëïöüÿâêîôûãõ@ñ"

        # Act
        result = _replace_elements(input_string, elem_list=custom_element_list)

        # Assert
        assert result == expected_output


class TestUnidecodeStrings:
    def test_unidecode_strings_with_default_characters_to_replace(self):
        # Arrange
        input_string = "Thérè ís (Sómé) Text: Here. It's A - Test; <With> ?Special/Characters, & More___Underscores"
        expected_output = (
            "there_is_some_text_here_it_s_a_test_with>_special_characters___more_underscores"
        )

        # Act
        result = _unidecode_strings(input_string)

        # Assert
        assert result == expected_output

    def test_unidecode_strings_with_custom_characters_to_replace(self):
        # Arrange
        input_string = "Thérè ís (Sómé) Text: Here. It's A - Test; <With> ?Special/Characters, & More___Underscores"
        custom_characters_to_replace = ["(", ")", "*", ":", "."]
        expected_output = "there is _some_ text_ here_ it's a - test; <with> ?special/characters, & more___underscores"

        # Act
        result = _unidecode_strings(
            input_string, characters_to_replace=custom_characters_to_replace
        )

        # Assert
        assert result == expected_output
