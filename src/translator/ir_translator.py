"""Strategy to IR translator.

Re-exports the typed visitor-based implementation.
"""

from .errors import TranslationError
from .translator import IRTranslator

__all__ = ["IRTranslator", "TranslationError"]
