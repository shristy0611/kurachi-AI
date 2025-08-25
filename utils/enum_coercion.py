"""
Centralized enum coercion utilities for safe enum conversion.

This module provides utilities for safely converting string values to enum members
with comprehensive error handling and case-insensitive matching.
"""

from enum import Enum
from typing import Union, TypeVar, Type

T = TypeVar('T', bound=Enum)


def as_enum(value: Union[str, Enum], enum_cls: Type[T]) -> T:
    """
    Safely coerce string or enum values to enum members.
    
    This function handles case-insensitive matching against both .name and .value
    attributes of enum members, providing robust conversion with helpful error messages.
    
    Args:
        value: String or enum value to convert
        enum_cls: Target enum class to convert to
        
    Returns:
        Enum member of the specified class
        
    Raises:
        ValueError: When the value cannot be matched to any enum member
        TypeError: When value is neither string nor enum, or enum_cls is not an Enum
        
    Examples:
        >>> from services.translation_service import Language
        >>> as_enum("en", Language)
        <Language.ENGLISH: 'en'>
        >>> as_enum("ENGLISH", Language)
        <Language.ENGLISH: 'en'>
        >>> as_enum(Language.ENGLISH, Language)
        <Language.ENGLISH: 'en'>
    """
    # Type validation
    if not isinstance(enum_cls, type) or not issubclass(enum_cls, Enum):
        raise TypeError(f"enum_cls must be an Enum class, got {type(enum_cls)}")
    
    # If already the correct enum type, return as-is
    if isinstance(value, enum_cls):
        return value
    
    # If it's a different enum type, raise error
    if isinstance(value, Enum):
        raise TypeError(f"Cannot convert {type(value)} to {enum_cls}")
    
    # Must be a string at this point
    if not isinstance(value, str):
        raise TypeError(f"Value must be string or {enum_cls.__name__}, got {type(value)}")
    
    # Normalize the input string
    normalized_value = value.strip().lower()
    
    # Try exact match first (case-insensitive)
    for member in enum_cls:
        # Check against enum value (e.g., "en", "ja")
        if member.value.lower() == normalized_value:
            return member
        
        # Check against enum name (e.g., "ENGLISH", "JAPANESE")
        if member.name.lower() == normalized_value:
            return member
    
    # If no match found, provide helpful error message
    valid_values = []
    for member in enum_cls:
        valid_values.extend([member.value, member.name.lower()])
    
    valid_values_str = ", ".join(f"'{v}'" for v in sorted(set(valid_values)))
    
    raise ValueError(
        f"Invalid {enum_cls.__name__} value: '{value}'. "
        f"Valid options are: {valid_values_str}"
    )


def get_enum_key_safe(enum_member: Enum) -> str:
    """
    Get a safe string key for enum members to use in dictionaries.
    
    This ensures consistent hashable keys by using the enum's name attribute.
    
    Args:
        enum_member: Enum member to get key for
        
    Returns:
        String key based on enum name
        
    Examples:
        >>> from services.translation_service import Language
        >>> get_enum_key_safe(Language.ENGLISH)
        'ENGLISH'
    """
    if not isinstance(enum_member, Enum):
        raise TypeError(f"Expected Enum member, got {type(enum_member)}")
    
    return enum_member.name


def get_enum_value_safe(enum_member: Enum) -> str:
    """
    Get the enum value safely for serialization or external APIs.
    
    Args:
        enum_member: Enum member to get value for
        
    Returns:
        String value of the enum member
        
    Examples:
        >>> from services.translation_service import Language
        >>> get_enum_value_safe(Language.ENGLISH)
        'en'
    """
    if not isinstance(enum_member, Enum):
        raise TypeError(f"Expected Enum member, got {type(enum_member)}")
    
    return enum_member.value