from typing import List, Dict

TILE_NAMES = [
    "1w", "2w", "3w", "4w", "5w", "6w", "7w", "8w", "9w",
    "1t", "2t", "3t", "4t", "5t", "6t", "7t", "8t", "9t",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "red",
]
TILE_COUNT = len(TILE_NAMES)
RED_TILE_ID = TILE_NAMES.index("red")
TILE_NAME_TO_ID = {name: idx for idx, name in enumerate(TILE_NAMES)}

def tile_to_name(tile: int) -> str:
    if tile == -1:
        return "G"
    if 0 <= tile < TILE_COUNT:
        return TILE_NAMES[tile]
    return "?"
