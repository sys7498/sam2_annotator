# 사용 가이드

## 1. 설치와 실행

Ubuntu WSL2 + RTX A6000에서 최초 한 번만 실행한다.

```bash
git clone https://github.com/sys7498/sam2_annotator.git
cd sam2_annotator
bash scripts/setup_wsl_a6000.sh
```

설치가 끝나면 다음 중 하나로 연다.

```bash
conda activate sam2-annotator

# 파일 탐색기에서 하나 선택
python desktop_annotator.py --pick

# 영상 하나를 경로로 지정
python desktop_annotator.py --input /mnt/c/path/to/video.mp4

# 폴더의 영상 목록을 TODO/DONE 상태와 함께 표시
python desktop_annotator.py --dataset /mnt/c/path/to/videos
```

지원 입력은 MP4, AVI, MOV, MKV, WebM, MPEG 계열 영상과 이미지 또는 frame 이미지 폴더다.
결과는 입력과 분리되어 `outputs/<입력_이름>/` 아래에 저장된다.

`--dataset`은 하위 폴더를 재귀적으로 탐색한다. 따라서
`raw/<사람>/<작업>/rgb.mp4`처럼 중간 폴더가 추가돼도 자동으로 찾고,
결과도 `outputs/<사람>/<작업>/rgb/`로 분리되어 같은 `rgb.mp4` 파일명이 충돌하지 않는다.

## Aria raw 영상 5fps 프레임 추출

사람 폴더 하나의 MP4를 재귀적으로 찾아 5fps JPEG 프레임 시퀀스로 만든다. 원본의 중간 폴더 구조를 보존한다.

```bash
# 먼저 어떤 파일이 처리될지 확인
bash scripts/sample_aria_recording_5fps.sh jdh --dry-run

# raw/jdh/**/rgb.mp4 → 5fps_sampled/jdh/**/rgb/000000.jpg
bash scripts/sample_aria_recording_5fps.sh jdh

# 이미 만든 파일도 다시 생성
bash scripts/sample_aria_recording_5fps.sh jdh --overwrite

# raw/ 아래 모든 사람 폴더 변환
bash scripts/sample_aria_recording_5fps.sh --all
```

기본 경로는 `/workspace/shared/aria_recording/`이며, 다른 위치에서는 아래처럼 지정한다.

```bash
ARIA_RECORDING_ROOT=/다른/aria_recording \
  bash scripts/sample_aria_recording_5fps.sh jdh
```

macOS metadata 파일(`._rgb.mp4`)은 자동으로 건너뛴다. JPEG 품질은 FFmpeg `-q:v 2`이며,
audio는 추출하지 않는다. 중단된 작업은 완성된 `rgb/` 폴더를 건너뛰고 나머지부터 재개한다.

## 2. 기본 작업 흐름

1. 첫 프레임에서 `n`으로 새 object ID를 만든다.
2. 대상 내부를 좌클릭하거나, 대상을 감싸서 좌드래그한다.
3. 필요하면 대상 밖/원치 않는 부분을 우클릭해 negative point를 추가한다.
4. `Space`로 다음 프레임으로 이동한다. SAM2가 현재 object를 다음 프레임으로 전파한다.
5. 구조 변화나 tracking 오류가 보이는 프레임에서 prompt 또는 brush로 보정한다.
6. `s`로 중간 저장한다. `q` 또는 창 닫기 시에도 자동 저장된다.

화면의 mask 색, 윤곽선, `id:<번호>` 라벨로 현재 추적 상태를 확인한다. 활성 ID에는 `*`가 붙는다.

## 3. 마우스 조작

| 조작 | 동작 |
| --- | --- |
| 좌클릭 | 활성 object ID에 positive point 추가 |
| 좌드래그 | 활성 object ID에 box prompt 적용 |
| 우클릭 | 활성 object ID에 negative point 추가 |
| Shift + 좌클릭/좌드래그 | hand ID에 positive point/box prompt |
| Shift + 우클릭 | hand ID에 negative point |
| 편집 모드에서 좌드래그 | mask 영역 추가 |
| 편집 모드에서 우드래그 | mask 영역 제거 |

점과 box는 object 또는 hand ID가 없으면 해당 종류의 ID를 자동으로 만든다. box는 해당 ID의 이전
점 prompt를 대체한다.

## 4. 키보드 단축키

| 키 | 동작 |
| --- | --- |
| `Space`, `.` | 다음 프레임으로 전파 |
| `[` / `]` | 이전 / 다음 활성 ID 선택 |
| `n` | 새 object ID 생성 |
| `h`, `Shift+n` | 새 hand ID 생성 |
| `g` | 번호를 입력해 ID 선택 또는 새 ID 생성 |
| `d` | 번호를 입력해 ID 하나 이상 삭제 |
| `c`, `r` | 활성 ID 번호 변경 |
| `x` | 활성 ID의 현재 프레임 point prompt 초기화 |
| `p` | 터미널 prompt 편집기 열기 (`+ x y`, `- x y`, `clear`, `apply`) |
| `v` | 현재 ID·면적·prompt 상태를 터미널에 표시 |
| `e` | brush mask 편집 모드 시작/취소 |
| `a` | brush 보정 mask를 SAM2 상태에 적용 |
| `z` | brush 보정을 원본 mask로 되돌림 |
| `+` / `-` | brush 크기 증가 / 감소 |
| `l` | 자동 생성된 predecessor–successor 관계 출력 |
| `s` | 즉시 저장 |
| `t` | 활성 object의 투명 배경 crop PNG 저장 |
| `o` | 현재 프레임의 outline PNG 저장 |
| `q`, `Esc` | 저장 후 종료 |

## 5. 구조 변화 관계 자동 기록

가림이나 일시적 tracking 실패와 실제 분리를 구분하기 위해, relation은 ID lifecycle을 명시적으로
표시했을 때만 자동 기록된다.

분리와 결합은 별도 relation 입력 창 없이 평소 ID lifecycle을 따라 자동 기록된다.

| 변화 | 마스크 작업 | 자동 relation |
| --- | --- | --- |
| 하나가 둘로 분리 | 부모 ID 하나를 `d`로 종료 → `n`으로 새 object ID 두 개 생성 → 각 ID에 prompt | `separation`: `1→2` |
| 둘이 하나로 결합 | 부모 ID 두 개를 `d`에서 쉼표로 함께 종료 → `n`으로 새 object ID 하나 생성 → prompt | `joining`: `2→1` |

새 ID는 부모 ID 종료 시점 전후 12프레임 안에서 생성돼야 한다. 그러면
`lineage_relations.json`에 predecessor IDs, successor IDs, transition frame과 `type`이 기록된다.
생성 뒤 successor를 삭제하면 해당 relation은 `needs_review`가 되므로 확인 후 다시 지정한다.
`l`로 현재 relation을 터미널에서 확인한다. 가림이나 tracking 실패에는 `d`를 누르지 않으면
relation이 생성되지 않는다.

## 6. 저장 결과

각 입력은 다음 경로에 저장된다.

```text
outputs/<입력_이름>/
├── annotations_ytvis.json
├── interactive_session_meta.json
├── lineage_relations.json
├── transparent_crops/<영상_이름>/  # t를 눌렀을 때
└── outline_only/<영상_이름>/        # o를 눌렀을 때
```

- `annotations_ytvis.json`: track ID별 COCO-style uncompressed RLE mask, bbox, area
- `interactive_session_meta.json`: 처리한 프레임, 영상 입력과 결과 파일 위치
- `lineage_relations.json`: 자동 생성된 separation/joining predecessor–successor 관계 및 검수 상태

## 7. WSL 문제 해결

| 증상 | 확인/해결 |
| --- | --- |
| `CUDA is unavailable` | Windows NVIDIA WSL 드라이버를 설치/갱신한 뒤 WSL에서 `nvidia-smi` 확인 |
| OpenCV 창 또는 파일 선택창이 안 뜸 | Windows 11 WSLg를 활성화하거나 X 서버를 실행. `echo $DISPLAY` 확인 |
| `/mnt/c` 영상을 못 찾음 | Windows 드라이브 마운트 상태를 확인하고 절대 경로를 사용 |
| checkpoint 오류 | 저장소 루트에서 `bash scripts/setup_wsl_a6000.sh`를 다시 실행 |
| 긴 영상에서 GPU 메모리 부족 | `python desktop_annotator.py --input ... --state-window 96` 사용. 과거 프레임 수정은 재전파가 필요 |

설치 상태만 다시 점검할 때는 아래를 실행한다.

```bash
conda run -n sam2-annotator python scripts/verify_a6000.py
```
