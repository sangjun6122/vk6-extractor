# VK6 Extractor

Keyence VK6 레이저 현미경 파일에서 높이맵과 이미지를 추출하는 Python 도구입니다.

## 기능

- **32비트 높이맵** 추출 (TIFF, Halcon 호환)
- **16비트 높이맵** 추출 (TIFF)
- **레이저 이미지** 추출 (PNG)
- **광학 이미지** 추출 (PNG)
- **썸네일 이미지** 추출 (PNG)
- **측정 정보** 추출 (TXT)

## 요구사항

```bash
pip install numpy pillow
```

## 사용법

### 모든 VK6 파일 일괄 추출

```bash
python extract_vk6.py
```

`extracted/` 폴더에 다음과 같이 정리됩니다:
```
extracted/
├── height/      # 32비트 높이맵 (um 단위)
├── height16/    # 16비트 높이맵
├── laser/       # 레이저 이미지
├── optical/     # 광학 이미지
├── thumbnail/   # 썸네일 이미지
└── info/        # 측정 정보
```

### 특정 파일만 추출

```bash
python extract_vk6.py sample.vk6
```

현재 폴더에 `sample_height.tiff`, `sample_laser.png` 등으로 저장됩니다.

## 출력 포맷

| 파일 | 설명 | 포맷 |
|------|------|------|
| `*_height.tiff` | 32비트 높이맵 (um 단위) | 32-bit float TIFF |
| `*_height16.tiff` | 16비트 높이맵 | 16-bit TIFF |
| `*_laser.png` | 레이저 이미지 | RGB PNG |
| `*_optical.png` | 광학 이미지 | RGB PNG |
| `*_thumb*.png` | 썸네일 이미지 | RGB PNG |
| `*_info.txt` | 측정 정보 | Text |

## Halcon에서 사용

```
read_image(HeightMap, 'extracted/height/1_height.tiff')
```

## 측정 정보 예시

```
측정 일시: 2025/12/10 12:07:27
대물렌즈 배율: 28.1x
줌: 1.3x
원본 이미지 크기: 7462 x 7270
X 피치: 271085 nm (271.085 um)
Y 피치: 271085 nm (271.085 um)
Z 피치: 54168 nm (54.168 um)
```

## 지원 파일

- Keyence VK-X series (.vk6)
- 내부적으로 VK4 포맷 사용

## 라이선스

MIT License
