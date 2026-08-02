import json

INPUT_PATH = "cards_by_id.json"
OUTPUT_PATH = "cards.json"


def clean_name(raw_name):
    """Nimmt nur den ersten (englischen) Teil vor dem ersten ' / ' -
    der Rest ist japanisch/chinesisch und wird von der App nicht gebraucht."""
    if not raw_name:
        return ""
    return raw_name.split(" / ")[0].strip()


def join_list(value):
    """colors/attributes/types kommen als Liste - CardEntity erwartet einen
    String, getrennt mit '/', genau wie es der bestehende Scanner-Code schon erwartet."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "/".join(str(v) for v in value)
    return str(value)


def derive_pack_id(card_id):
    """pack_id in den Rohdaten ist eine interne Nummer, nicht der Set-Code.
    Der echte Set-Code steckt als Präfix in der Karten-ID selbst (z.B. 'ST01' aus 'ST01-001')."""
    if card_id and "-" in card_id:
        return card_id.split("-")[0]
    return ""


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        raw_cards = json.load(f)

    app_cards = []
    skipped = 0

    for card_id, card in raw_cards.items():
        if not card.get("id"):
            skipped += 1
            continue

        app_cards.append({
            "id": card.get("id"),
            "name": clean_name(card.get("name")),
            "category": card.get("category") or "",
            "colors": join_list(card.get("colors")),
            "attributes": join_list(card.get("attributes")),
            "types": join_list(card.get("types")),
            "cost": card.get("cost") if card.get("cost") is not None else 0,
            "power": card.get("power"),
            "counter": card.get("counter"),
            "effect": card.get("effect") or "",
            "trigger": card.get("trigger"),
            "rarity": card.get("rarity") or "",
            "packId": derive_pack_id(card.get("id")),
            "imgUrl": card.get("img_full_url") or "",
        })

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(app_cards, f, ensure_ascii=False)

    print(f"Fertig: {len(app_cards)} Karten geschrieben nach {OUTPUT_PATH}")
    if skipped:
        print(f"({skipped} Einträge übersprungen, keine gültige ID)")

    if app_cards:
        print("\nBeispiel (erste Karte, zur Kontrolle):")
        print(json.dumps(app_cards[0], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()