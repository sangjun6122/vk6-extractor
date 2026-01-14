#!/usr/bin/env python3
"""
Extract height maps and images from Keyence VK6 laser microscope files.
- Height map: 32-bit float TIFF (Halcon compatible)
- Laser image: PNG
- Optical image: PNG
"""

import struct
import zipfile
import io
import os
import sys
import numpy as np
from PIL import Image


def parse_vk6(filepath):
    """Parse VK6 file and extract height map, images, and measurement info."""

    with open(filepath, 'rb') as f:
        data = f.read()

    # VK6 file consists of BMP preview image + ZIP compressed data
    pk_pos = data.find(b'PK')
    if pk_pos == -1:
        raise ValueError("ZIP data not found")

    zip_data = data[pk_pos:]

    with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
        vk4_data = zf.read('Vk4File')

    # Parse VK4 file
    if vk4_data[:4] != b'VK4_':
        raise ValueError("VK4 magic not found")

    # Read offset table (starting from offset 16)
    offsets = []
    for i in range(16, 80, 4):
        val = struct.unpack('<I', vk4_data[i:i+4])[0]
        if val > 0:
            offsets.append((i, val))

    result = {
        'laser_image': None,      # Laser image (first color)
        'optical_image': None,    # Optical image (second color)
        'height_data': None,      # 32-bit height data
        'height_16bit': None,     # 16-bit height data
        'thumbnails': [],         # Thumbnail images
        'height_width': 0,
        'height_height': 0,
        'image_width': 0,
        'image_height': 0,
        'z_scale': 1.0,
        'measure_info': {}        # Measurement info
    }

    # Parse measurement info
    result['measure_info'] = {
        'year': struct.unpack('<I', vk4_data[88:92])[0],
        'month': struct.unpack('<I', vk4_data[92:96])[0],
        'day': struct.unpack('<I', vk4_data[96:100])[0],
        'hour': struct.unpack('<I', vk4_data[100:104])[0],
        'minute': struct.unpack('<I', vk4_data[104:108])[0],
        'second': struct.unpack('<I', vk4_data[108:112])[0],
        'objective_mag': struct.unpack('<I', vk4_data[148:152])[0] / 1000.0,  # Objective magnification
        'zoom': struct.unpack('<I', vk4_data[152:156])[0] / 100.0,  # Zoom
        'orig_width': struct.unpack('<I', vk4_data[188:192])[0],  # Original image width
        'orig_height': struct.unpack('<I', vk4_data[192:196])[0],  # Original image height
        'x_pitch_nm': struct.unpack('<I', vk4_data[252:256])[0],  # X pitch (nm)
        'y_pitch_nm': struct.unpack('<I', vk4_data[256:260])[0],  # Y pitch (nm)
        'z_pitch_nm': struct.unpack('<I', vk4_data[280:284])[0],  # Z pitch (nm)
    }

    color_images = []  # High resolution color images
    thumbnail_images = []  # Thumbnail images

    # Analyze each block
    for header_pos, offset in offsets:
        if offset + 20 > len(vk4_data):
            continue

        width = struct.unpack('<I', vk4_data[offset:offset+4])[0]
        height = struct.unpack('<I', vk4_data[offset+4:offset+8])[0]
        bits = struct.unpack('<I', vk4_data[offset+8:offset+12])[0]
        marker = vk4_data[offset+12:offset+16]
        data_size = struct.unpack('<I', vk4_data[offset+16:offset+20])[0]

        if width == 0 or height == 0 or width > 10000 or height > 10000:
            continue

        # 32-bit height data (MC32 marker)
        if marker == b'MC32' and bits == 32:
            header_size = 796
            data_start = offset + header_size

            if data_start + data_size <= len(vk4_data):
                height_bytes = vk4_data[data_start:data_start + data_size]
                height_array = np.frombuffer(height_bytes, dtype=np.int32).reshape((height, width))
                result['height_data'] = height_array.copy()
                result['height_width'] = width
                result['height_height'] = height

                z_scale = struct.unpack('<I', vk4_data[offset+20:offset+24])[0]
                if z_scale > 0:
                    result['z_scale'] = z_scale / 1000000.0

        # 16-bit height data (MC16 marker)
        elif marker == b'MC16' and bits == 16:
            header_size = 796
            data_start = offset + header_size

            if data_start + data_size <= len(vk4_data):
                height_bytes = vk4_data[data_start:data_start + data_size]
                height_array = np.frombuffer(height_bytes, dtype=np.uint16).reshape((height, width))
                result['height_16bit'] = height_array.copy()

        # 24-bit color image
        elif bits == 24:
            if data_size == width * height * 3:
                data_start = offset + 20
                if data_start + data_size <= len(vk4_data):
                    rgb_bytes = vk4_data[data_start:data_start + data_size]
                    rgb_array = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((height, width, 3))
                    rgb_array = rgb_array[:, :, ::-1].copy()  # BGR -> RGB

                    if width >= 1024 and height >= 768:
                        # High resolution image
                        color_images.append({
                            'data': rgb_array,
                            'width': width,
                            'height': height,
                            'header_pos': header_pos
                        })
                    else:
                        # Thumbnail image
                        thumbnail_images.append({
                            'data': rgb_array,
                            'width': width,
                            'height': height,
                            'header_pos': header_pos
                        })

    # Assign color images (by header position: first=laser, second=optical)
    color_images.sort(key=lambda x: x['header_pos'])
    if len(color_images) >= 1:
        result['laser_image'] = color_images[0]['data']
        result['image_width'] = color_images[0]['width']
        result['image_height'] = color_images[0]['height']
    if len(color_images) >= 2:
        result['optical_image'] = color_images[1]['data']

    # Assign thumbnail images
    thumbnail_images.sort(key=lambda x: x['header_pos'])
    result['thumbnails'] = [t['data'] for t in thumbnail_images]

    return result


def save_height_tiff(height_data, output_path, z_scale=1.0):
    """Save height data as 32-bit float TIFF (Halcon compatible)."""
    # Convert 0.1nm to um (raw data is in 0.1nm units)
    height_float = height_data.astype(np.float32) / 10000.0  # 0.1nm to um

    # Save as 32-bit float TIFF using PIL
    img = Image.fromarray(height_float, mode='F')
    img.save(output_path, format='TIFF')


def save_height_16bit_tiff(height_data, output_path):
    """Save 16-bit height data as TIFF."""
    img = Image.fromarray(height_data.astype(np.uint16))
    img.save(output_path, format='TIFF')


def save_color_image(color_data, output_path):
    """Save color image as PNG."""
    img = Image.fromarray(color_data, mode='RGB')
    img.save(output_path, format='PNG')


def save_measure_info(info, output_path):
    """Save measurement info as text file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"Measurement Date: {info['year']}/{info['month']:02d}/{info['day']:02d} {info['hour']:02d}:{info['minute']:02d}:{info['second']:02d}\n")
        f.write(f"Objective Magnification: {info['objective_mag']:.1f}x\n")
        f.write(f"Zoom: {info['zoom']:.1f}x\n")
        f.write(f"Original Image Size: {info['orig_width']} x {info['orig_height']}\n")
        f.write(f"X Pitch: {info['x_pitch_nm']} nm ({info['x_pitch_nm']/1000:.3f} um)\n")
        f.write(f"Y Pitch: {info['y_pitch_nm']} nm ({info['y_pitch_nm']/1000:.3f} um)\n")
        f.write(f"Z Pitch: {info['z_pitch_nm']} nm ({info['z_pitch_nm']/1000:.3f} um)\n")


def save_height_csv(height_data, output_path, info, basename):
    """Save height data as CSV with metadata header (Keyence compatible format)."""
    # Convert 0.1nm to um
    height_um = height_data.astype(np.float64) / 10000.0

    # Calculate min/max (excluding invalid values where raw data is 0)
    valid_mask = height_data != 0
    if np.any(valid_mask):
        min_val = height_um[valid_mask].min()
        max_val = height_um[valid_mask].max()
    else:
        min_val = 0.0
        max_val = 0.0

    height_h, height_w = height_data.shape

    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        # Write metadata header
        f.write(f'"Measured date","{info["year"]}-{info["month"]:02d}-{info["day"]:02d} {info["hour"]:02d}:{info["minute"]:02d}:{info["second"]:02d}"\n')
        f.write('"Model","VK-X3000 Series"\n')
        f.write('"Data type\\n","ImageDataCsv"\n')
        f.write('"File version","1000"\n')
        f.write(f'"Measurement data name","{basename}"\n')
        f.write('"Resolution","Standard"\n')
        f.write('"Measurement Mode","Surface profile"\n')
        f.write('"Scan Mode","Laser confocal"\n')
        f.write(f'"Objective Lens Power","{int(info["objective_mag"])}"\n')
        f.write(f'"XY Calibration","{info["x_pitch_nm"]/1000:.3f}","nm"\n')
        f.write('"Output image data","Height"\n')
        f.write(f'"Horizontal","{height_w}"\n')
        f.write(f'"Vertical","{height_h}"\n')
        f.write(f'"Minimum value","{min_val:.3f}"\n')
        f.write(f'"Maximum value","{max_val:.3f}"\n')
        f.write('"Unit","um"\n')
        f.write('"Reference data name",""\n')
        f.write('\n')
        f.write('"Height"\n')

        # Write height data row by row
        for row in height_um:
            row_str = ','.join(f'"{v:.3f}"' for v in row)
            f.write(row_str + '\n')


def process_vk6_file(vk6_path, output_dir=None):
    """Process VK6 file - extract all data (skip if not available)."""
    if output_dir is None:
        output_dir = os.path.dirname(vk6_path)

    basename = os.path.splitext(os.path.basename(vk6_path))[0]

    print(f"Processing: {vk6_path}")

    try:
        result = parse_vk6(vk6_path)

        # Save measurement info
        info = result['measure_info']
        info_path = os.path.join(output_dir, f"{basename}_info.txt")
        save_measure_info(info, info_path)
        print(f"  Measurement info: {info['year']}/{info['month']:02d}/{info['day']:02d}, {info['objective_mag']:.1f}x, pitch {info['x_pitch_nm']/1000:.3f}um")

        # Save laser image
        if result['laser_image'] is not None:
            laser_path = os.path.join(output_dir, f"{basename}_laser.png")
            save_color_image(result['laser_image'], laser_path)
            print(f"  Laser image: {result['image_width']}x{result['image_height']}")

        # Save optical image
        if result['optical_image'] is not None:
            optical_path = os.path.join(output_dir, f"{basename}_optical.png")
            save_color_image(result['optical_image'], optical_path)
            print(f"  Optical image: {result['image_width']}x{result['image_height']}")

        # Save 16-bit height map
        if result['height_16bit'] is not None:
            height16_path = os.path.join(output_dir, f"{basename}_height16.tiff")
            save_height_16bit_tiff(result['height_16bit'], height16_path)
            print(f"  16-bit height map: {result['height_16bit'].min()}~{result['height_16bit'].max()}")

        # Save 32-bit height map
        if result['height_data'] is not None:
            height_path = os.path.join(output_dir, f"{basename}_height.tiff")
            save_height_tiff(result['height_data'], height_path, result['z_scale'])
            print(f"  32-bit height map: {result['height_width']}x{result['height_height']}, {result['height_data'].min()/10000:.2f}~{result['height_data'].max()/10000:.2f} um")

            # Save CSV format
            csv_path = os.path.join(output_dir, f"{basename}_height.csv")
            save_height_csv(result['height_data'], csv_path, info, basename)
            print(f"  CSV height map: {csv_path}")

        # Save thumbnail images
        for i, thumb in enumerate(result['thumbnails']):
            thumb_path = os.path.join(output_dir, f"{basename}_thumb{i+1}.png")
            save_color_image(thumb, thumb_path)
        if result['thumbnails']:
            print(f"  Thumbnails: {len(result['thumbnails'])}")

        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def process_vk6_file_organized(vk6_path, base_output_dir):
    """Process VK6 file - save organized by folders."""
    basename = os.path.splitext(os.path.basename(vk6_path))[0]

    print(f"Processing: {vk6_path}")

    try:
        result = parse_vk6(vk6_path)

        # Save measurement info
        info = result['measure_info']
        info_path = os.path.join(base_output_dir, 'info', f"{basename}_info.txt")
        save_measure_info(info, info_path)
        print(f"  Measurement info: {info['year']}/{info['month']:02d}/{info['day']:02d}, {info['objective_mag']:.1f}x, pitch {info['x_pitch_nm']/1000:.3f}um")

        # Save laser image
        if result['laser_image'] is not None:
            laser_path = os.path.join(base_output_dir, 'laser', f"{basename}_laser.png")
            save_color_image(result['laser_image'], laser_path)
            print(f"  Laser image: {result['image_width']}x{result['image_height']}")

        # Save optical image
        if result['optical_image'] is not None:
            optical_path = os.path.join(base_output_dir, 'optical', f"{basename}_optical.png")
            save_color_image(result['optical_image'], optical_path)
            print(f"  Optical image: {result['image_width']}x{result['image_height']}")

        # Save 16-bit height map
        if result['height_16bit'] is not None:
            height16_path = os.path.join(base_output_dir, 'height16', f"{basename}_height16.tiff")
            save_height_16bit_tiff(result['height_16bit'], height16_path)
            print(f"  16-bit height map: {result['height_16bit'].min()}~{result['height_16bit'].max()}")

        # Save 32-bit height map
        if result['height_data'] is not None:
            height_path = os.path.join(base_output_dir, 'height', f"{basename}_height.tiff")
            save_height_tiff(result['height_data'], height_path, result['z_scale'])
            print(f"  32-bit height map: {result['height_width']}x{result['height_height']}, {result['height_data'].min()/10000:.2f}~{result['height_data'].max()/10000:.2f} um")

            # Save CSV format
            csv_path = os.path.join(base_output_dir, 'height_csv', f"{basename}_height.csv")
            save_height_csv(result['height_data'], csv_path, info, basename)
            print(f"  CSV height map: {csv_path}")

        # Save thumbnail images
        for i, thumb in enumerate(result['thumbnails']):
            thumb_path = os.path.join(base_output_dir, 'thumbnail', f"{basename}_thumb{i+1}.png")
            save_color_image(thumb, thumb_path)
        if result['thumbnails']:
            print(f"  Thumbnails: {len(result['thumbnails'])}")

        return True

    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    """Main function."""
    vk6_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) < 2:
        # Process all VK6 files in current directory (organized by folders)
        vk6_files = sorted([f for f in os.listdir(vk6_dir) if f.endswith('.vk6')])

        if not vk6_files:
            print("No VK6 files found.")
            return

        # Create output folders
        output_dir = os.path.join(vk6_dir, 'extracted')
        for subdir in ['height', 'height16', 'height_csv', 'laser', 'optical', 'thumbnail', 'info']:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

        print(f"Processing {len(vk6_files)} VK6 files.")
        print(f"Output folder: {output_dir}\n")

        success = 0
        for vk6_file in vk6_files:
            vk6_path = os.path.join(vk6_dir, vk6_file)
            if process_vk6_file_organized(vk6_path, output_dir):
                success += 1
            print()

        print(f"Done: {success}/{len(vk6_files)} files processed")
    else:
        # Process specified files (save to current folder)
        for vk6_path in sys.argv[1:]:
            process_vk6_file(vk6_path)


if __name__ == '__main__':
    main()
