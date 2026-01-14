# VK6 Extractor

A Python tool for extracting height maps and images from Keyence VK6 laser microscope files.

## Features

- **32-bit height map** extraction (TIFF, Halcon compatible)
- **16-bit height map** extraction (TIFF)
- **Laser image** extraction (PNG)
- **Optical image** extraction (PNG)
- **Thumbnail image** extraction (PNG)
- **Measurement info** extraction (TXT)

## Requirements

```bash
pip install numpy pillow
```

## Usage

### Batch extraction of all VK6 files

```bash
python extract_vk6.py
```

Outputs are organized into the `extracted/` folder:
```
extracted/
├── height/      # 32-bit height maps (um units, TIFF)
├── height16/    # 16-bit height maps (TIFF)
├── height_csv/  # Height maps with metadata (CSV, Keyence format)
├── laser/       # Laser images
├── optical/     # Optical images
├── thumbnail/   # Thumbnail images
└── info/        # Measurement info
```

### Extract specific file

```bash
python extract_vk6.py sample.vk6
```

Saves as `sample_height.tiff`, `sample_laser.png`, etc. in the current folder.

## Output Format

| File | Description | Format |
|------|-------------|--------|
| `*_height.tiff` | 32-bit height map (um units) | 32-bit float TIFF |
| `*_height16.tiff` | 16-bit height map | 16-bit TIFF |
| `*_height.csv` | Height map with metadata | CSV (Keyence format) |
| `*_laser.png` | Laser image | RGB PNG |
| `*_optical.png` | Optical image | RGB PNG |
| `*_thumb*.png` | Thumbnail images | RGB PNG |
| `*_info.txt` | Measurement info | Text |

## Halcon Usage

```
read_image(HeightMap, 'extracted/height/1_height.tiff')
```

## Measurement Info Example

```
Measurement Date: 2025/12/10 12:07:27
Objective Magnification: 28.1x
Zoom: 1.3x
Original Image Size: 7462 x 7270
X Pitch: 271085 nm (271.085 um)
Y Pitch: 271085 nm (271.085 um)
Z Pitch: 54168 nm (54.168 um)
```

## Supported Files

- Keyence VK-X series (.vk6)
- Uses VK4 format internally

## License

MIT License
