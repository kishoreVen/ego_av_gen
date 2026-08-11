"""
Download a small set of glTF sample models to use as table objects.
All models are from the official KhronosGroup/glTF-Sample-Models repo (CC0 / public domain).
Run from the experiment root:  python download_assets.py
"""

import pathlib
import urllib.request
import urllib.error

ASSETS_DIR = pathlib.Path(__file__).parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)

BASE = "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Models/main/2.0"

MODELS = [
    ("Duck",           "Duck/glTF-Binary/Duck.glb"),
    ("Avocado",        "Avocado/glTF-Binary/Avocado.glb"),
    ("ToyCar",         "ToyCar/glTF-Binary/ToyCar.glb"),
    ("Lantern",        "Lantern/glTF-Binary/Lantern.glb"),
    ("WaterBottle",    "WaterBottle/glTF-Binary/WaterBottle.glb"),
    ("DamagedHelmet",  "DamagedHelmet/glTF-Binary/DamagedHelmet.glb"),
    ("BarramundiFish", "BarramundiFish/glTF-Binary/BarramundiFish.glb"),
    ("BoomBox",        "BoomBox/glTF-Binary/BoomBox.glb"),
    ("AntiqueCamera",  "AntiqueCamera/glTF-Binary/AntiqueCamera.glb"),
    ("Suzanne",        "Suzanne/glTF-Binary/Suzanne.glb"),
    ("FlightHelmet",   "FlightHelmet/glTF-Binary/FlightHelmet.glb"),
    ("Fox",            "Fox/glTF-Binary/Fox.glb"),
    ("DragonAttenuation", "DragonAttenuation/glTF-Binary/DragonAttenuation.glb"),
    ("StainedGlassLamp",  "StainedGlassLamp/glTF-Binary/StainedGlassLamp.glb"),
    ("IridescenceLamp",   "IridescenceLamp/glTF-Binary/IridescenceLamp.glb"),
    ("CesiumMilkTruck",   "CesiumMilkTruck/glTF-Binary/CesiumMilkTruck.glb"),
]


def download(name: str, path: str) -> None:
    url  = f"{BASE}/{path}"
    dest = ASSETS_DIR / f"{name}.glb"
    if dest.exists():
        print(f"  {name}.glb  already exists, skipping")
        return
    print(f"  downloading {name}.glb …", end=" ", flush=True)
    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size // 1024
        print(f"{size_kb} KB")
    except urllib.error.HTTPError as e:
        print(f"FAILED ({e.code})")
    except Exception as e:
        print(f"FAILED ({e})")


if __name__ == "__main__":
    print(f"Saving assets to: {ASSETS_DIR}\n")
    for name, path in MODELS:
        download(name, path)
    print("\nDone.  Missing files will use coloured-primitive fallbacks in the scene.")
