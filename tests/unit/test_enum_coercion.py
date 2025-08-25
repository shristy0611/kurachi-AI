"""
Comprehensive tests for enum coercion utilities.

Tests cover valid inputs, case variations, error conditions, and edge cases
for the centralized enum coercion system.
"""

import pytest
from enum import Enum
from typing import Any

from utils.enum_coercion import as_enum, get_enum_key_safe, get_enum_value_safe
from services.translation_service import Language
from services.multilingual_conversation_interface import UILanguage, ResponseLanguage


class SampleEnum(Enum):
    """Sample enum for validation testing"""
    OPTION_A = "a"
    OPTION_B = "b"
    COMPLEX_NAME = "complex-value"


class TestAsEnum:
    """Test cases for as_enum function"""
    
    @pytest.mark.parametrize("input_value,expected", [
        # Test with Language enum - exact value matches
        ("en", Language.ENGLISH),
        ("ja", Language.JAPANESE),
        ("es", Language.SPANISH),
        ("fr", Language.FRENCH),
        ("de", Language.GERMAN),
        ("zh", Language.CHINESE),
        ("ko", Language.KOREAN),
        ("auto", Language.AUTO),
        
        # Test with Language enum - exact name matches (case-insensitive)
        ("english", Language.ENGLISH),
        ("japanese", Language.JAPANESE),
        ("spanish", Language.SPANISH),
        ("french", Language.FRENCH),
        ("german", Language.GERMAN),
        ("chinese", Language.CHINESE),
        ("korean", Language.KOREAN),
        
        # Test case variations
        ("EN", Language.ENGLISH),
        ("En", Language.ENGLISH),
        ("ENGLISH", Language.ENGLISH),
        ("English", Language.ENGLISH),
        ("JAPANESE", Language.JAPANESE),
        ("Japanese", Language.JAPANESE),
        
        # Test with whitespace
        (" en ", Language.ENGLISH),
        (" ENGLISH ", Language.ENGLISH),
        ("  ja  ", Language.JAPANESE),
        
        # Test with UILanguage enum
        ("en", UILanguage.ENGLISH),
        ("ja", UILanguage.JAPANESE),
        ("english", UILanguage.ENGLISH),
        ("ENGLISH", UILanguage.ENGLISH),
        
        # Test with ResponseLanguage enum
        ("auto", ResponseLanguage.AUTO),
        ("original", ResponseLanguage.ORIGINAL),
        ("AUTO", ResponseLanguage.AUTO),
        ("ORIGINAL", ResponseLanguage.ORIGINAL),
        
        # Test with custom SampleEnum
        ("a", SampleEnum.OPTION_A),
        ("b", SampleEnum.OPTION_B),
        ("complex-value", SampleEnum.COMPLEX_NAME),
        ("option_a", SampleEnum.OPTION_A),
        ("OPTION_A", SampleEnum.OPTION_A),
        ("complex_name", SampleEnum.COMPLEX_NAME),
        ("COMPLEX_NAME", SampleEnum.COMPLEX_NAME),
    ])
    def test_valid_string_inputs(self, input_value: str, expected: Enum):
        """Test that valid string inputs are correctly converted to enum members"""
        result = as_enum(input_value, type(expected))
        assert result == expected
        assert isinstance(result, type(expected))
    
    @pytest.mark.parametrize("enum_value", [
        Language.ENGLISH,
        Language.JAPANESE,
        Language.SPANISH,
        UILanguage.ENGLISH,
        UILanguage.JAPANESE,
        ResponseLanguage.AUTO,
        ResponseLanguage.ORIGINAL,
        SampleEnum.OPTION_A,
        SampleEnum.OPTION_B,
    ])
    def test_enum_passthrough(self, enum_value: Enum):
        """Test that existing enum values are returned unchanged"""
        result = as_enum(enum_value, type(enum_value))
        assert result is enum_value  # Should be the exact same object
        assert result == enum_value
    
    @pytest.mark.parametrize("invalid_value,enum_cls", [
        # Invalid language codes
        ("invalid", Language),
        ("xyz", Language),
        ("", Language),
        ("eng", Language),  # Close but not exact
        ("jp", Language),   # Close but not exact
        
        # Invalid UI language codes
        ("es", UILanguage),  # Spanish not supported in UILanguage
        ("invalid", UILanguage),
        
        # Invalid response language codes
        ("es", ResponseLanguage),
        ("manual", ResponseLanguage),
        
        # Invalid sample enum values
        ("c", SampleEnum),
        ("option_c", SampleEnum),
        ("invalid", SampleEnum),
    ])
    def test_invalid_string_inputs(self, invalid_value: str, enum_cls: type):
        """Test that invalid string inputs raise ValueError with helpful messages"""
        with pytest.raises(ValueError) as exc_info:
            as_enum(invalid_value, enum_cls)
        
        error_msg = str(exc_info.value)
        assert f"Invalid {enum_cls.__name__} value: '{invalid_value}'" in error_msg
        assert "Valid options are:" in error_msg
        
        # Verify that valid options are listed in the error message
        for member in enum_cls:
            assert member.value in error_msg or member.name.lower() in error_msg
    
    @pytest.mark.parametrize("invalid_input", [
        123,
        12.34,
        [],
        {},
        None,
        object(),
    ])
    def test_invalid_input_types(self, invalid_input: Any):
        """Test that non-string, non-enum inputs raise TypeError"""
        with pytest.raises(TypeError) as exc_info:
            as_enum(invalid_input, Language)
        
        error_msg = str(exc_info.value)
        assert "Value must be string or Language" in error_msg
        assert str(type(invalid_input)) in error_msg
    
    def test_wrong_enum_type_conversion(self):
        """Test that converting between different enum types raises TypeError"""
        ui_lang = UILanguage.ENGLISH
        
        with pytest.raises(TypeError) as exc_info:
            as_enum(ui_lang, Language)
        
        error_msg = str(exc_info.value)
        assert "Cannot convert" in error_msg
        assert "UILanguage" in error_msg
        assert "Language" in error_msg
    
    @pytest.mark.parametrize("invalid_enum_cls", [
        str,
        int,
        list,
        dict,
        object,
        "not_a_class",
        123,
    ])
    def test_invalid_enum_class(self, invalid_enum_cls: Any):
        """Test that non-Enum classes raise TypeError"""
        with pytest.raises(TypeError) as exc_info:
            as_enum("test", invalid_enum_cls)
        
        error_msg = str(exc_info.value)
        assert "enum_cls must be an Enum class" in error_msg


class TestGetEnumKeySafe:
    """Test cases for get_enum_key_safe function"""
    
    @pytest.mark.parametrize("enum_member,expected_key", [
        (Language.ENGLISH, "ENGLISH"),
        (Language.JAPANESE, "JAPANESE"),
        (Language.AUTO, "AUTO"),
        (UILanguage.ENGLISH, "ENGLISH"),
        (UILanguage.JAPANESE, "JAPANESE"),
        (ResponseLanguage.AUTO, "AUTO"),
        (ResponseLanguage.ORIGINAL, "ORIGINAL"),
        (SampleEnum.OPTION_A, "OPTION_A"),
        (SampleEnum.COMPLEX_NAME, "COMPLEX_NAME"),
    ])
    def test_valid_enum_keys(self, enum_member: Enum, expected_key: str):
        """Test that enum members return correct name-based keys"""
        result = get_enum_key_safe(enum_member)
        assert result == expected_key
        assert isinstance(result, str)
    
    @pytest.mark.parametrize("invalid_input", [
        "string",
        123,
        [],
        {},
        None,
        object(),
    ])
    def test_invalid_input_types(self, invalid_input: Any):
        """Test that non-enum inputs raise TypeError"""
        with pytest.raises(TypeError) as exc_info:
            get_enum_key_safe(invalid_input)
        
        error_msg = str(exc_info.value)
        assert "Expected Enum member" in error_msg
        assert str(type(invalid_input)) in error_msg


class TestGetEnumValueSafe:
    """Test cases for get_enum_value_safe function"""
    
    @pytest.mark.parametrize("enum_member,expected_value", [
        (Language.ENGLISH, "en"),
        (Language.JAPANESE, "ja"),
        (Language.SPANISH, "es"),
        (Language.AUTO, "auto"),
        (UILanguage.ENGLISH, "en"),
        (UILanguage.JAPANESE, "ja"),
        (ResponseLanguage.AUTO, "auto"),
        (ResponseLanguage.ORIGINAL, "original"),
        (SampleEnum.OPTION_A, "a"),
        (SampleEnum.OPTION_B, "b"),
        (SampleEnum.COMPLEX_NAME, "complex-value"),
    ])
    def test_valid_enum_values(self, enum_member: Enum, expected_value: str):
        """Test that enum members return correct values"""
        result = get_enum_value_safe(enum_member)
        assert result == expected_value
        assert isinstance(result, str)
    
    @pytest.mark.parametrize("invalid_input", [
        "string",
        123,
        [],
        {},
        None,
        object(),
    ])
    def test_invalid_input_types(self, invalid_input: Any):
        """Test that non-enum inputs raise TypeError"""
        with pytest.raises(TypeError) as exc_info:
            get_enum_value_safe(invalid_input)
        
        error_msg = str(exc_info.value)
        assert "Expected Enum member" in error_msg
        assert str(type(invalid_input)) in error_msg


class TestEnumCoercionIntegration:
    """Integration tests for enum coercion utilities"""
    
    def test_dictionary_key_usage(self):
        """Test that enum keys work properly in dictionaries"""
        # Test that we can use enum members as dictionary keys safely
        lang_dict = {}
        
        # Using get_enum_key_safe for consistent keys
        english_key = get_enum_key_safe(Language.ENGLISH)
        japanese_key = get_enum_key_safe(Language.JAPANESE)
        
        lang_dict[english_key] = "English content"
        lang_dict[japanese_key] = "Japanese content"
        
        assert lang_dict[english_key] == "English content"
        assert lang_dict[japanese_key] == "Japanese content"
        assert len(lang_dict) == 2
    
    def test_round_trip_conversion(self):
        """Test that string -> enum -> string conversion works correctly"""
        test_cases = [
            ("en", Language),
            ("japanese", Language),
            ("ENGLISH", Language),
            ("auto", ResponseLanguage),
            ("original", ResponseLanguage),
        ]
        
        for input_str, enum_cls in test_cases:
            # Convert string to enum
            enum_member = as_enum(input_str, enum_cls)
            
            # Convert back to string value
            output_value = get_enum_value_safe(enum_member)
            
            # The output should be the canonical enum value
            assert output_value == enum_member.value
            
            # And we should be able to convert back
            round_trip_enum = as_enum(output_value, enum_cls)
            assert round_trip_enum == enum_member
    
    def test_case_insensitive_consistency(self):
        """Test that different case variations all resolve to the same enum"""
        variations = ["en", "EN", "En", "english", "ENGLISH", "English"]
        
        results = [as_enum(var, Language) for var in variations]
        
        # All should resolve to the same enum member
        assert all(result == Language.ENGLISH for result in results)
        
        # All should have the same key and value
        keys = [get_enum_key_safe(result) for result in results]
        values = [get_enum_value_safe(result) for result in results]
        
        assert all(key == "ENGLISH" for key in keys)
        assert all(value == "en" for value in values)