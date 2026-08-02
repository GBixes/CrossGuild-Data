import json
import os
import urllib.request
from io import BytesIO
from PIL import Image
import torch
import torchvision.models as models
from torchvision import transforms

print("Lade Deep-Learning-Modell (PyTorch MobileNetV3Large Feature Extractor)...")
weights = models.MobileNet_V3_Large_Weights.DEFAULT
model = models.mobilenet_v3_large(weights=weights)
model.eval()
print("Modell erfolgreich initialisiert!\n")

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def process_image_to_vector(image_bytes):
    try:
        img = Image.open(BytesIO(image_bytes)).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0)

        with torch.no_grad():
            x = model.features(img_tensor)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
            return x_norm[0].tolist()
    except Exception as e:
        print(f" -> Fehler bei der Bildverarbeitung: {e}")
        return None

def main():
    input_json_path = "cards_by_id.json"
    output_json_path = "card_embeddings.json"
    cache_dir = "downloaded_images"

    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    if not os.path.exists(input_json_path):
        print(f"FEHLER: Datei '{input_json_path}' wurde im Ordner nicht gefunden!")
        return

    with open(input_json_path, "r", encoding="utf-8") as f:
        cards_data = json.load(f)

    card_items = []
    if isinstance(cards_data, dict):
        for key, value in cards_data.items():
            if isinstance(value, dict):
                if "id" not in value or not value["id"]:
                    value["id"] = str(key)
                card_items.append(value)
    elif isinstance(cards_data, list):
        card_items = cards_data

    # NEU: Bestehende Embeddings laden statt bei 0 anzufangen -
    # dadurch verarbeiten wir nur wirklich neue Karten.
    embeddings_database = {}
    if os.path.exists(output_json_path):
        with open(output_json_path, "r", encoding="utf-8") as f:
            embeddings_database = json.load(f)
        print(f"Bestehende Datei geladen: {len(embeddings_database)} Karten bereits vorhanden.\n")

    already_done_count = len(embeddings_database)
    total_cards = len(card_items)
    new_count = 0
    skipped_count = 0

    print(f"Prüfe {total_cards} Karten aus '{input_json_path}'...\n")

    possible_url_keys = ["img_full_url", "imgUrl", "image_url", "img_url", "image", "img", "url", "cardImage", "src", "card_url", "artwork"]

    for index, card in enumerate(card_items):
        card_id = str(card.get("id", "")).strip()

        if not card_id:
            skipped_count += 1
            continue

        # NEU: Schon vorhanden? Überspringen - spart Downloads und Rechenzeit.
        if card_id in embeddings_database:
            continue

        img_url = ""
        for url_key in possible_url_keys:
            val = card.get(url_key, "")
            if val and isinstance(val, str) and val.startswith("http"):
                img_url = val.strip()
                break

        if not img_url:
            print(f"[{index + 1}/{total_cards}] Übersprungen ({card_id}): Keine gültige HTTP-URL gefunden.")
            skipped_count += 1
            continue

        print(f"[{index + 1}/{total_cards}] NEU - Verarbeite: {card_id}...")

        safe_filename = card_id.replace("/", "_").replace("\\", "_") + ".jpg"
        local_img_path = os.path.join(cache_dir, safe_filename)

        image_bytes = None
        if os.path.exists(local_img_path):
            try:
                with open(local_img_path, "rb") as img_file:
                    image_bytes = img_file.read()
            except Exception:
                pass

        if not image_bytes:
            try:
                req = urllib.request.Request(img_url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req, timeout=10) as response:
                    image_bytes = response.read()
                with open(local_img_path, "wb") as img_file:
                    img_file.write(image_bytes)
            except Exception as e:
                print(f" -> Download fehlerhaft für {card_id} ({img_url}): {e}")
                skipped_count += 1
                continue

        if image_bytes:
            vector = process_image_to_vector(image_bytes)
            if vector is not None:
                embeddings_database[card_id] = vector
                new_count += 1
            else:
                skipped_count += 1

    print("\n" + "="*50)
    print(f"ABSCHLUSS-BERICHT:")
    print(f" -> Bereits vorhanden (übersprungen): {already_done_count}")
    print(f" -> Neu hinzugefügt: {new_count}")
    print(f" -> Übersprungen / fehlerhaft: {skipped_count}")
    print(f" -> Gesamt in der Datei jetzt: {len(embeddings_database)}")
    print("="*50)

    if new_count > 0:
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(embeddings_database, f)
        print(f"\nERFOLG! Datei aktualisiert: {output_json_path}")
        print(f"{new_count} neue Karten hinzugefügt - alte Karten unverändert übernommen.")
    else:
        print("\nKeine neuen Karten gefunden - Datei wurde nicht verändert.")

if __name__ == "__main__":
    main()