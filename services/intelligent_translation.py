"""
Compatibility shim for legacy intelligent_translation interface.
Provides glossary loading and context-aware translate_with_context that
proxies to the SOTA orchestrator where appropriate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional
import os
from pathlib import Path

import yaml

from services.sota_translation_orchestrator import (
    sota_translation_orchestrator,
    Language,
    TranslationQuality,
    TranslationRequest,
)


@dataclass
class QualityScore:
    overall_score: float = 0.9


@dataclass
class IntelligentTranslationResult:
    translated_text: str
    quality_score: QualityScore


@dataclass
class IntelligentContext:
    domain: str = "general"
    style: str = "business"
    audience: str = "general"

# Backward-compatible alias used by some tests
TranslationContext = IntelligentContext


class GlossaryManager:
    def __init__(self) -> None:
        self.glossaries: Dict[str, Dict[str, str]] = {}
        self._load_all()

    def _normalize_glossary_entries(self, entries: Dict[str, str]) -> Dict[str, str]:
        normalized: Dict[str, str] = {}
        for k, v in entries.items():
            key = str(k).strip()
            val = str(v).strip()
            if key and val:
                normalized[key] = val
        return normalized

    def _load_yaml(self, path: Path) -> Dict[str, str]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return self._normalize_glossary_entries(data)
            return {}
        except Exception:
            return {}

    def _load_all(self) -> None:
        base = Path("config/glossaries")
        mapping = {
            "business": base / "business.yml",
            "technical": base / "technical.yml",
        }
        for name, path in mapping.items():
            if path.exists():
                self.glossaries[name] = self._load_yaml(path)
            else:
                self.glossaries[name] = {}

    def get_glossary(self, name: str) -> Dict[str, str]:
        return self.glossaries.get(name, {})


class IntelligentTranslationService:
    def __init__(self) -> None:
        self.glossary_manager = GlossaryManager()

    def translate_with_context(
        self,
        text: str,
        target_language: str,
        source_language: str = "auto",
        context: Optional[IntelligentContext] = None,
    ) -> IntelligentTranslationResult:
        # Map strings to orchestrator enums
        def to_lang(code: str) -> Language:
            for m in Language:
                if m.value == code or m.name.lower() == code.lower():
                    return m
            return Language.AUTO

        tgt = to_lang(target_language)
        src = to_lang(source_language)
        quality = TranslationQuality.BUSINESS

        req = TranslationRequest(
            text=text,
            source_lang=src,
            target_lang=tgt,
            quality=quality,
            context=str(context) if context else None,
        )
        result = sota_translation_orchestrator.translate(
            req.text, req.target_lang, req.source_lang, req.quality, req.context
        )
        return IntelligentTranslationResult(
            translated_text=result.translated_text,
            quality_score=QualityScore(overall_score=getattr(result, "confidence", 0.9)),
        )


__all__ = [
    "IntelligentTranslationService",
    "GlossaryManager",
    "IntelligentContext",
    "TranslationContext",
    "IntelligentTranslationResult",
]
