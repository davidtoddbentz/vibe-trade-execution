"""Archetype expansion: converts specialized archetypes to primitive rule_trigger forms.

This module implements the "Archetypes as Validated Compositions" pattern where
specialized archetypes (entry.trend_pullback, exit.band_exit, etc.) are expanded
to their primitive forms (entry.rule_trigger, exit.rule_trigger) using ConditionSpec
composition.

Benefits:
- Single translation path for each primitive type
- Schema-driven extensibility (new archetypes = new templates, not code)
- Preserved semantic meaning for agents
- Consistent condition compilation
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ArchetypeExpander:
    """Expands specialized archetypes to primitive rule_trigger forms."""

    def __init__(self, templates_path: Path | None = None):
        """Initialize with expansion templates.

        Args:
            templates_path: Path to archetype_expansions.json. If None, uses default location.
        """
        if templates_path is None:
            # Default path relative to this file
            templates_path = (
                Path(__file__).parent.parent.parent.parent
                / "vibe-trade-mcp"
                / "data"
                / "archetype_expansions.json"
            )

        self._templates: dict[str, Any] = {}
        self._load_templates(templates_path)

    def _load_templates(self, path: Path) -> None:
        """Load expansion templates from JSON file."""
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                # Flatten entry/exit/gate/overlay archetypes into single lookup
                self._templates = {}
                for category in ["entry_archetypes", "exit_archetypes", "gate_archetypes", "overlay_archetypes"]:
                    if category in data:
                        self._templates.update(data[category])
        else:
            # No templates file - all archetypes treated as primitives
            self._templates = {}

    def is_primitive(self, archetype: str) -> bool:
        """Check if an archetype is a primitive (doesn't need expansion)."""
        if archetype not in self._templates:
            return True  # Unknown archetypes are treated as primitives
        template = self._templates[archetype]
        return template.get("is_primitive", False) or template.get("expands_to") is None

    def get_primitive_type(self, archetype: str) -> str:
        """Get the primitive type an archetype expands to."""
        if self.is_primitive(archetype):
            return archetype
        return self._templates[archetype]["expands_to"]

    def expand(self, card: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Expand a card to its primitive form.

        Args:
            card: The card to expand (must have 'type_id' and 'slots')

        Returns:
            Tuple of (expanded_card, provenance). If no expansion needed,
            returns (original_card, None).
        """
        archetype = card.get("type_id", "")

        if self.is_primitive(archetype):
            return card, None

        template = self._templates[archetype]
        expansion = template.get("expansion_template", {})
        inference_rules = template.get("inference_rules", {})

        # Deep copy the card to avoid mutations
        expanded = copy.deepcopy(card)
        expanded["type_id"] = template["expands_to"]

        # Get the slots from the original card
        slots = card.get("slots", {})

        # Build the expanded event structure
        if "event" in expansion:
            expanded_event = self._substitute_slots(expansion["event"], slots, card, inference_rules)
            expanded["slots"]["event"] = expanded_event

        # Build provenance for debugging
        provenance = {
            "source_archetype": archetype,
            "expanded_to": template["expands_to"],
            "expanded_at": datetime.now(timezone.utc).isoformat(),
            "template_version": "1.0"
        }

        return expanded, provenance

    def _substitute_slots(
        self,
        template: Any,
        slots: dict[str, Any],
        card: dict[str, Any],
        inference_rules: dict[str, Any] | None = None
    ) -> Any:
        """Recursively substitute slot references in a template.

        Supports:
        - $slot:path - Substitute value from slots at path
        - $const:value - Use constant value
        - $infer:key - Infer from inference rules based on action.direction or mode
        - $negate:path - Negate numeric value from path
        """
        if inference_rules is None:
            inference_rules = {}

        if isinstance(template, str):
            if template.startswith("$slot:"):
                path = template[6:]  # Remove "$slot:" prefix
                return self._get_path(slots, path)
            elif template.startswith("$const:"):
                value_str = template[7:]
                # Try to parse as number
                try:
                    return int(value_str)
                except ValueError:
                    try:
                        return float(value_str)
                    except ValueError:
                        return value_str
            elif template.startswith("$negate:"):
                path = template[8:]
                value = self._get_path(slots, path)
                if isinstance(value, (int, float)):
                    return -value
                return value
            elif template.startswith("$infer:"):
                # Inference uses rules based on action.direction or event mode
                rule_key = template[7:]  # Remove "$infer:" prefix
                if rule_key in inference_rules:
                    rule_map = inference_rules[rule_key]
                    # Try action.direction first
                    direction = self._get_path(slots, "action.direction")
                    if direction and direction in rule_map:
                        return rule_map[direction]
                    # Try event mode (for gap_play)
                    mode = self._get_path(slots, "event.session.mode")
                    if mode and mode in rule_map:
                        return rule_map[mode]
                    # Return first value as default
                    if rule_map:
                        return next(iter(rule_map.values()))
                return None
            return template

        elif isinstance(template, dict):
            result = {}
            for key, value in template.items():
                substituted = self._substitute_slots(value, slots, card, inference_rules)
                # Only include non-None values
                if substituted is not None:
                    result[key] = substituted
            return result

        elif isinstance(template, list):
            return [self._substitute_slots(item, slots, card, inference_rules) for item in template]

        return template

    def _get_path(self, obj: dict[str, Any], path: str) -> Any:
        """Get a value from a nested dict using dot-notation path."""
        parts = path.split(".")
        current = obj
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current


def expand_card(card: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Convenience function to expand a single card.

    Args:
        card: The card to expand

    Returns:
        Tuple of (expanded_card, provenance)
    """
    expander = ArchetypeExpander()
    return expander.expand(card)
