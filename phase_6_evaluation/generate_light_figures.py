from pathlib import Path

from PIL import Image


FIGURE_NAMES = [
    "accuracy_comparison_bar.png",
    "accuracy_summary_table.png",
    "per_image_f1_scatter.png",
    "rl_imgsz_distribution.png",
    "sysparams_bar_comparison.png",
    "sysparams_boxplots.png",
    "sysparams_radar.png",
    "sysparams_summary_table.png",
    "sysparams_timeseries.png",
]

SOURCE_DIR = Path(__file__).resolve().parent / "output"


def convert_pixel(pixel: tuple[int, ...]) -> tuple[int, ...]:
    red, green, blue, *alpha = pixel

    # Replace the dark neutral palette while preserving colored data series.
    if max(red, green, blue) < 55 and max(red, green, blue) - min(red, green, blue) < 24:
        red, green, blue = 255, 255, 255
    elif max(red, green, blue) < 90 and max(red, green, blue) - min(red, green, blue) < 28:
        red, green, blue = 245, 245, 245
    elif min(red, green, blue) > 185 and max(red, green, blue) - min(red, green, blue) < 35:
        red, green, blue = 35, 35, 35

    return (red, green, blue, *alpha)


def main() -> None:
    for name in FIGURE_NAMES:
        source_path = SOURCE_DIR / name
        if not source_path.exists():
            continue

        image = Image.open(source_path).convert("RGBA")
        pixels = image.load()
        for y in range(image.height):
            for x in range(image.width):
                pixels[x, y] = convert_pixel(pixels[x, y])

        output_path = SOURCE_DIR / name.replace(".png", "_light.png")
        image.save(output_path, optimize=True)
        print(f"[SAVED] {output_path.name}")


if __name__ == "__main__":
    main()