"""
Generate data/annotations.json from the data/images/ folder structure.

Dataset layout expected:
    data/images/
        Seahorse/img1.jpg
        Sharks/img1.jpg
        ...

Since there are no bounding-box labels, the entire image is used as the
detection region (bbox = [0, 0, width, height]).  This lets the model learn
species identity from whole-image context; box regression will tighten once
proper bounding-box annotations are added later.

Usage:
    python train/generate_annotations.py
    python train/generate_annotations.py --dataset-dir data/images --out data/annotations.json
"""
import argparse
import json
import os

from PIL import Image

# ── Taxonomy map ──────────────────────────────────────────────────────────────
# (phylum, class_, order, family, species)
TAXONOMY: dict[str, tuple[str, str, str, str, str]] = {
    "Clams":           ("Mollusca",     "Bivalvia",      "Veneroida",         "Veneridae",          "Clams"),
    "Corals":          ("Cnidaria",     "Anthozoa",      "Scleractinia",      "Acroporidae",        "Corals"),
    "Crabs":           ("Arthropoda",   "Malacostraca",  "Decapoda",          "Portunidae",         "Crabs"),
    "Dolphin":         ("Chordata",     "Mammalia",      "Artiodactyla",      "Delphinidae",        "Dolphin"),
    "Eel":             ("Chordata",     "Actinopterygii","Anguilliformes",    "Anguillidae",        "Eel"),
    "Fish":            ("Chordata",     "Actinopterygii","Perciformes",       "Labridae",           "Fish"),
    "Jelly Fish":      ("Cnidaria",     "Scyphozoa",     "Semaeostomeae",     "Ulmaridae",          "Jelly Fish"),
    "Lobster":         ("Arthropoda",   "Malacostraca",  "Decapoda",          "Nephropidae",        "Lobster"),
    "Nudibranchs":     ("Mollusca",     "Gastropoda",    "Nudibranchia",      "Chromodorididae",    "Nudibranchs"),
    "Octopus":         ("Mollusca",     "Cephalopoda",   "Octopoda",          "Octopodidae",        "Octopus"),
    "Otter":           ("Chordata",     "Mammalia",      "Carnivora",         "Mustelidae",         "Otter"),
    "Penguin":         ("Chordata",     "Aves",          "Sphenisciformes",   "Spheniscidae",       "Penguin"),
    "Puffers":         ("Chordata",     "Actinopterygii","Tetraodontiformes", "Tetraodontidae",     "Puffers"),
    "Sea Rays":        ("Chordata",     "Chondrichthyes","Myliobatiformes",   "Dasyatidae",         "Sea Rays"),
    "Sea Urchins":     ("Echinodermata","Echinoidea",    "Camarodonta",       "Strongylocentrotidae","Sea Urchins"),
    "Seahorse":        ("Chordata",     "Actinopterygii","Syngnathiformes",   "Syngnathidae",       "Seahorse"),
    "Seal":            ("Chordata",     "Mammalia",      "Carnivora",         "Phocidae",           "Seal"),
    "Sharks":          ("Chordata",     "Chondrichthyes","Carcharhiniformes", "Carcharhinidae",     "Sharks"),
    "Shrimp":          ("Arthropoda",   "Malacostraca",  "Decapoda",          "Penaeidae",          "Shrimp"),
    "Squid":           ("Mollusca",     "Cephalopoda",   "Teuthida",          "Loliginidae",        "Squid"),
    "Starfish":        ("Echinodermata","Asteroidea",    "Forcipulatida",     "Asteriidae",         "Starfish"),
    "Turtle_Tortoise": ("Chordata",     "Reptilia",      "Testudines",        "Cheloniidae",        "Turtle_Tortoise"),
    "Whale":           ("Chordata",     "Mammalia",      "Artiodactyla",      "Balaenopteridae",    "Whale"),
}

_SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _image_size(path: str) -> tuple[int, int]:
    """Return (width, height) without decoding pixel data."""
    with Image.open(path) as img:
        return img.size  # (width, height)


def generate(dataset_dir: str, out_path: str) -> None:
    # Collect all taxonomy vocabulary from the map
    levels = ["phylum", "class_", "order", "family", "species"]
    vocab: dict[str, set[str]] = {lvl: set() for lvl in levels}
    for tax in TAXONOMY.values():
        for lvl, val in zip(levels, tax):
            vocab[lvl].add(val)

    taxonomy_labels = {lvl: sorted(vocab[lvl]) for lvl in levels}

    images_list = []
    annotations_list = []
    img_id = 0
    ann_id = 0
    skipped = 0

    species_dirs = sorted(os.listdir(dataset_dir))
    for species_name in species_dirs:
        species_path = os.path.join(dataset_dir, species_name)
        if not os.path.isdir(species_path):
            continue

        if species_name not in TAXONOMY:
            print(f"  SKIP (no taxonomy mapping): {species_name}")
            skipped += 1
            continue

        tax = TAXONOMY[species_name]
        taxonomy_entry = dict(zip(levels, tax))

        for fname in sorted(os.listdir(species_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _SUPPORTED_EXTS:
                continue

            fpath = os.path.join(species_path, fname)
            try:
                w, h = _image_size(fpath)
            except Exception as e:
                print(f"  WARN could not read {fpath}: {e}")
                continue

            img_id += 1
            # Store path relative to dataset_dir so image_dir arg stays clean
            rel_path = os.path.join(species_name, fname)

            images_list.append({
                "id":        img_id,
                "file_name": rel_path,
                "width":     w,
                "height":    h,
            })

            ann_id += 1
            annotations_list.append({
                "id":       ann_id,
                "image_id": img_id,
                "bbox":     [0, 0, w, h],   # full-image box (no GT boxes available)
                "taxonomy": taxonomy_entry,
            })

    out = {
        "taxonomy_labels": taxonomy_labels,
        "images":          images_list,
        "annotations":     annotations_list,
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(
        f"\nWrote {len(images_list)} images / {len(annotations_list)} annotations "
        f"across {len(species_dirs) - skipped} species → {out_path}"
    )
    if skipped:
        print(f"Skipped {skipped} folder(s) with no taxonomy mapping.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate annotations.json from dataset/ folder")
    p.add_argument("--dataset-dir", default="data/images",    help="Root folder with species subfolders")
    p.add_argument("--out",         default="data/annotations.json", help="Output annotation file")
    args = p.parse_args()
    generate(args.dataset_dir, args.out)
