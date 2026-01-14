#!/usr/bin/env python3
"""
Keyence VK6 파일에서 높이맵과 이미지들을 추출하는 스크립트
- 높이맵: 32-bit float TIFF (Halcon 호환)
- 레이저 이미지: PNG
- 광학 이미지: PNG
"""

import struct
import zipfile
import io
import os
import sys
import numpy as np
from PIL import Image


def parse_vk6(filepath):
    """VK6 파일을 파싱하여 높이맵, 이미지, 측정 정보를 추출"""

    with open(filepath, 'rb') as f:
        data = f.read()

    # VK6 파일은 BMP 미리보기 이미지 + ZIP 압축 데이터로 구성
    pk_pos = data.find(b'PK')
    if pk_pos == -1:
        raise ValueError("ZIP 데이터를 찾을 수 없습니다")

    zip_data = data[pk_pos:]

    with zipfile.ZipFile(io.BytesIO(zip_data), 'r') as zf:
        vk4_data = zf.read('Vk4File')

    # VK4 파일 파싱
    if vk4_data[:4] != b'VK4_':
        raise ValueError("VK4 magic을 찾을 수 없습니다")

    # 오프셋 테이블 읽기 (offset 16부터)
    offsets = []
    for i in range(16, 80, 4):
        val = struct.unpack('<I', vk4_data[i:i+4])[0]
        if val > 0:
            offsets.append((i, val))

    result = {
        'laser_image': None,      # 레이저 이미지 (첫 번째 컬러)
        'optical_image': None,    # 광학 이미지 (두 번째 컬러)
        'height_data': None,      # 32비트 높이 데이터
        'height_16bit': None,     # 16비트 높이 데이터
        'thumbnails': [],         # 썸네일 이미지들
        'height_width': 0,
        'height_height': 0,
        'image_width': 0,
        'image_height': 0,
        'z_scale': 1.0,
        'measure_info': {}        # 측정 정보
    }

    # 측정 정보 파싱
    result['measure_info'] = {
        'year': struct.unpack('<I', vk4_data[88:92])[0],
        'month': struct.unpack('<I', vk4_data[92:96])[0],
        'day': struct.unpack('<I', vk4_data[96:100])[0],
        'hour': struct.unpack('<I', vk4_data[100:104])[0],
        'minute': struct.unpack('<I', vk4_data[104:108])[0],
        'second': struct.unpack('<I', vk4_data[108:112])[0],
        'objective_mag': struct.unpack('<I', vk4_data[148:152])[0] / 1000.0,  # 대물렌즈 배율
        'zoom': struct.unpack('<I', vk4_data[152:156])[0] / 100.0,  # 줌
        'orig_width': struct.unpack('<I', vk4_data[188:192])[0],  # 원본 이미지 너비
        'orig_height': struct.unpack('<I', vk4_data[192:196])[0],  # 원본 이미지 높이
        'x_pitch_nm': struct.unpack('<I', vk4_data[252:256])[0],  # X 피치 (nm)
        'y_pitch_nm': struct.unpack('<I', vk4_data[256:260])[0],  # Y 피치 (nm)
        'z_pitch_nm': struct.unpack('<I', vk4_data[280:284])[0],  # Z 피치 (nm)
    }

    color_images = []  # 고해상도 컬러 이미지들
    thumbnail_images = []  # 썸네일 이미지들

    # 각 블록 분석
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

        # 32비트 높이 데이터 (MC32 마커)
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

        # 16비트 높이 데이터 (MC16 마커)
        elif marker == b'MC16' and bits == 16:
            header_size = 796
            data_start = offset + header_size

            if data_start + data_size <= len(vk4_data):
                height_bytes = vk4_data[data_start:data_start + data_size]
                height_array = np.frombuffer(height_bytes, dtype=np.uint16).reshape((height, width))
                result['height_16bit'] = height_array.copy()

        # 24비트 컬러 이미지
        elif bits == 24:
            if data_size == width * height * 3:
                data_start = offset + 20
                if data_start + data_size <= len(vk4_data):
                    rgb_bytes = vk4_data[data_start:data_start + data_size]
                    rgb_array = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((height, width, 3))
                    rgb_array = rgb_array[:, :, ::-1].copy()  # BGR -> RGB

                    if width >= 1024 and height >= 768:
                        # 고해상도 이미지
                        color_images.append({
                            'data': rgb_array,
                            'width': width,
                            'height': height,
                            'header_pos': header_pos
                        })
                    else:
                        # 썸네일 이미지
                        thumbnail_images.append({
                            'data': rgb_array,
                            'width': width,
                            'height': height,
                            'header_pos': header_pos
                        })

    # 컬러 이미지 할당 (header 위치 순서대로: 첫 번째=레이저, 두 번째=광학)
    color_images.sort(key=lambda x: x['header_pos'])
    if len(color_images) >= 1:
        result['laser_image'] = color_images[0]['data']
        result['image_width'] = color_images[0]['width']
        result['image_height'] = color_images[0]['height']
    if len(color_images) >= 2:
        result['optical_image'] = color_images[1]['data']

    # 썸네일 이미지 할당
    thumbnail_images.sort(key=lambda x: x['header_pos'])
    result['thumbnails'] = [t['data'] for t in thumbnail_images]

    return result


def save_height_tiff(height_data, output_path, z_scale=1.0):
    """높이 데이터를 32-bit float TIFF로 저장 (Halcon 호환)"""
    # nm 단위를 um으로 변환 (Halcon에서 사용하기 편리)
    height_float = height_data.astype(np.float32) / 1000.0  # nm to um

    # PIL로 32-bit float TIFF 저장
    img = Image.fromarray(height_float, mode='F')
    img.save(output_path, format='TIFF')


def save_height_16bit_tiff(height_data, output_path):
    """16비트 높이 데이터를 TIFF로 저장"""
    # uint16 데이터를 32비트로 변환하여 저장 (Halcon 호환성)
    img = Image.fromarray(height_data.astype(np.uint16))
    img.save(output_path, format='TIFF')


def save_color_image(color_data, output_path):
    """컬러 이미지를 PNG로 저장"""
    img = Image.fromarray(color_data, mode='RGB')
    img.save(output_path, format='PNG')


def save_measure_info(info, output_path):
    """측정 정보를 텍스트 파일로 저장"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"측정 일시: {info['year']}/{info['month']:02d}/{info['day']:02d} {info['hour']:02d}:{info['minute']:02d}:{info['second']:02d}\n")
        f.write(f"대물렌즈 배율: {info['objective_mag']:.1f}x\n")
        f.write(f"줌: {info['zoom']:.1f}x\n")
        f.write(f"원본 이미지 크기: {info['orig_width']} x {info['orig_height']}\n")
        f.write(f"X 피치: {info['x_pitch_nm']} nm ({info['x_pitch_nm']/1000:.3f} um)\n")
        f.write(f"Y 피치: {info['y_pitch_nm']} nm ({info['y_pitch_nm']/1000:.3f} um)\n")
        f.write(f"Z 피치: {info['z_pitch_nm']} nm ({info['z_pitch_nm']/1000:.3f} um)\n")


def process_vk6_file(vk6_path, output_dir=None):
    """VK6 파일 처리 - 모든 데이터 추출 (없는 데이터는 패스)"""
    if output_dir is None:
        output_dir = os.path.dirname(vk6_path)

    basename = os.path.splitext(os.path.basename(vk6_path))[0]

    print(f"처리 중: {vk6_path}")

    try:
        result = parse_vk6(vk6_path)

        # 측정 정보 저장
        info = result['measure_info']
        info_path = os.path.join(output_dir, f"{basename}_info.txt")
        save_measure_info(info, info_path)
        print(f"  측정 정보: {info['year']}/{info['month']:02d}/{info['day']:02d}, {info['objective_mag']:.1f}x, 피치 {info['x_pitch_nm']/1000:.3f}um")

        # 레이저 이미지 저장
        if result['laser_image'] is not None:
            laser_path = os.path.join(output_dir, f"{basename}_laser.png")
            save_color_image(result['laser_image'], laser_path)
            print(f"  레이저 이미지: {result['image_width']}x{result['image_height']}")

        # 광학 이미지 저장
        if result['optical_image'] is not None:
            optical_path = os.path.join(output_dir, f"{basename}_optical.png")
            save_color_image(result['optical_image'], optical_path)
            print(f"  광학 이미지: {result['image_width']}x{result['image_height']}")

        # 16비트 높이맵 저장
        if result['height_16bit'] is not None:
            height16_path = os.path.join(output_dir, f"{basename}_height16.tiff")
            save_height_16bit_tiff(result['height_16bit'], height16_path)
            print(f"  16비트 높이맵: {result['height_16bit'].min()}~{result['height_16bit'].max()}")

        # 32비트 높이맵 저장
        if result['height_data'] is not None:
            height_path = os.path.join(output_dir, f"{basename}_height.tiff")
            save_height_tiff(result['height_data'], height_path, result['z_scale'])
            print(f"  32비트 높이맵: {result['height_width']}x{result['height_height']}, {result['height_data'].min()/1000:.2f}~{result['height_data'].max()/1000:.2f} um")

        # 썸네일 이미지 저장
        for i, thumb in enumerate(result['thumbnails']):
            thumb_path = os.path.join(output_dir, f"{basename}_thumb{i+1}.png")
            save_color_image(thumb, thumb_path)
        if result['thumbnails']:
            print(f"  썸네일: {len(result['thumbnails'])}개")

        return True

    except Exception as e:
        print(f"  오류: {e}")
        return False


def process_vk6_file_organized(vk6_path, base_output_dir):
    """VK6 파일 처리 - 폴더별로 정리하여 저장"""
    basename = os.path.splitext(os.path.basename(vk6_path))[0]

    print(f"처리 중: {vk6_path}")

    try:
        result = parse_vk6(vk6_path)

        # 측정 정보 저장
        info = result['measure_info']
        info_path = os.path.join(base_output_dir, 'info', f"{basename}_info.txt")
        save_measure_info(info, info_path)
        print(f"  측정 정보: {info['year']}/{info['month']:02d}/{info['day']:02d}, {info['objective_mag']:.1f}x, 피치 {info['x_pitch_nm']/1000:.3f}um")

        # 레이저 이미지 저장
        if result['laser_image'] is not None:
            laser_path = os.path.join(base_output_dir, 'laser', f"{basename}_laser.png")
            save_color_image(result['laser_image'], laser_path)
            print(f"  레이저 이미지: {result['image_width']}x{result['image_height']}")

        # 광학 이미지 저장
        if result['optical_image'] is not None:
            optical_path = os.path.join(base_output_dir, 'optical', f"{basename}_optical.png")
            save_color_image(result['optical_image'], optical_path)
            print(f"  광학 이미지: {result['image_width']}x{result['image_height']}")

        # 16비트 높이맵 저장
        if result['height_16bit'] is not None:
            height16_path = os.path.join(base_output_dir, 'height16', f"{basename}_height16.tiff")
            save_height_16bit_tiff(result['height_16bit'], height16_path)
            print(f"  16비트 높이맵: {result['height_16bit'].min()}~{result['height_16bit'].max()}")

        # 32비트 높이맵 저장
        if result['height_data'] is not None:
            height_path = os.path.join(base_output_dir, 'height', f"{basename}_height.tiff")
            save_height_tiff(result['height_data'], height_path, result['z_scale'])
            print(f"  32비트 높이맵: {result['height_width']}x{result['height_height']}, {result['height_data'].min()/1000:.2f}~{result['height_data'].max()/1000:.2f} um")

        # 썸네일 이미지 저장
        for i, thumb in enumerate(result['thumbnails']):
            thumb_path = os.path.join(base_output_dir, 'thumbnail', f"{basename}_thumb{i+1}.png")
            save_color_image(thumb, thumb_path)
        if result['thumbnails']:
            print(f"  썸네일: {len(result['thumbnails'])}개")

        return True

    except Exception as e:
        print(f"  오류: {e}")
        return False


def main():
    """메인 함수"""
    vk6_dir = os.path.dirname(os.path.abspath(__file__))

    if len(sys.argv) < 2:
        # 현재 디렉토리의 모든 VK6 파일 처리 (폴더별 정리)
        vk6_files = sorted([f for f in os.listdir(vk6_dir) if f.endswith('.vk6')])

        if not vk6_files:
            print("VK6 파일을 찾을 수 없습니다.")
            return

        # 출력 폴더 생성
        output_dir = os.path.join(vk6_dir, 'extracted')
        for subdir in ['height', 'height16', 'laser', 'optical', 'thumbnail', 'info']:
            os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

        print(f"총 {len(vk6_files)}개의 VK6 파일을 처리합니다.")
        print(f"출력 폴더: {output_dir}\n")

        success = 0
        for vk6_file in vk6_files:
            vk6_path = os.path.join(vk6_dir, vk6_file)
            if process_vk6_file_organized(vk6_path, output_dir):
                success += 1
            print()

        print(f"완료: {success}/{len(vk6_files)} 파일 처리됨")
    else:
        # 지정된 파일들 처리 (현재 폴더에 저장)
        for vk6_path in sys.argv[1:]:
            process_vk6_file(vk6_path)


if __name__ == '__main__':
    main()
