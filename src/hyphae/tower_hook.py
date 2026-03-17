
# --- Tower integration hook ---
def check_tower_proximity(fact_text, fact_embedding=None):
    """After a fact is saved, check if it belongs to an existing tower.
    If so, add it as raw material. If enough materials, trigger synthesis.
    """
    try:
        from hyphae.towers import (
            get_all_towers, add_material, check_synthesis_needed,
            synthesize_floors, TOWER_PROXIMITY
        )
        import sqlite3
        import numpy as np

        towers = get_all_towers()
        if not towers:
            return

        # Simple keyword matching (fast, no embedding needed)
        text_lower = fact_text.lower()
        for tower in towers:
            tower_name_lower = tower["name"].lower()
            tower_purpose_lower = tower["purpose"].lower()

            # Check if fact mentions the tower name or key purpose words
            purpose_words = set(tower_purpose_lower.split()) - {"a", "the", "is", "and", "or", "for", "with", "of", "to", "in"}
            name_match = tower_name_lower in text_lower
            purpose_match = sum(1 for w in purpose_words if w in text_lower and len(w) > 3) >= 2

            if name_match or purpose_match:
                add_material(tower["id"], fact_text)
                if check_synthesis_needed(tower["id"]):
                    synthesize_floors(tower["id"])
    except Exception as e:
        pass  # Never block the remember flow
