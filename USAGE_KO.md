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

# 기본: /workspace/shared/aria_recording/5fps_sampled_sequences 목록 열기
python desktop_annotator.py

# 파일 탐색기에서 하나 선택
python desktop_annotator.py --pick

# 영상 하나를 경로로 지정
python desktop_annotator.py --input /mnt/c/path/to/video.mp4

# 기본 dataset 대신 다른 폴더의 영상 목록을 TODO/DONE 상태와 함께 표시
python desktop_annotator.py --dataset /mnt/c/path/to/videos
```

지원 입력은 MP4, AVI, MOV, MKV, WebM, MPEG 계열 영상과 이미지 또는 frame 이미지 폴더다.
기본 dataset은 `/workspace/shared/aria_recording/5fps_sampled_sequences`로 고정되어 있다. 다른 데이터셋이나
영상 하나를 열 때만 `--dataset` 또는 `--input`을 지정한다.
annotation 저장 루트는 `/workspace/shared/aria_recording/5fps_sampled_masks`로 고정되어 있다. 다른 경로가
필요한 경우에만 `--output-dir /path/to/annotations`로 덮어쓴다.
`--dataset` 실행에서는 입력 데이터셋의 하위 폴더 구조를 유지해
`/workspace/shared/aria_recording/5fps_sampled_masks/<사람>/<작업>/rgb/`에 결과를 저장한다.
결과는 입력과 분리되어 `/workspace/shared/aria_recording/5fps_sampled_masks/<입력_이름>/` 아래에 저장된다.

`--dataset`은 하위 폴더를 재귀적으로 탐색한다. 따라서
`raw/<사람>/<작업>/rgb.mp4`처럼 중간 폴더가 추가돼도 자동으로 찾고,
결과도 `/workspace/shared/aria_recording/5fps_sampled_masks/<사람>/<작업>/rgb/`로 분리되어 같은
`rgb.mp4` 파일명이 충돌하지 않는다.

## Aria raw 영상 5fps 프레임 추출

사람 폴더 하나의 MP4를 재귀적으로 찾아 5fps JPEG 프레임 시퀀스로 만든다. 원본의 중간 폴더 구조를 보존한다.

```bash
# 먼저 어떤 파일이 처리될지 확인
bash scripts/sample_aria_recording_5fps.sh jdh --dry-run

# raw/jdh/**/rgb.mp4 → 5fps_sampled_sequences/jdh/**/rgb/000000.jpg
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

## Aria 원본 FPS 전체 프레임 추출

5fps annotation용 시퀀스와 별도로, 원본의 모든 frame을 JPEG로 보관하려면 다음 스크립트를 쓴다.
영상의 실제 FPS에 따라 출력 폴더가 자동으로 정해진다. 현재 Aria raw 영상은 모두 30fps이므로
`raw/jdh/**/rgb.mp4`는 `30fps_raw/jdh/**/rgb/000000.jpg` 형태로 저장된다.

```bash
# 처리 대상을 먼저 확인
bash scripts/extract_aria_recording_native_fps.sh jdh --dry-run

# 사람 한 명의 모든 원본 frame 추출
bash scripts/extract_aria_recording_native_fps.sh jdh

# raw/ 아래 모든 사람의 원본 frame 추출
bash scripts/extract_aria_recording_native_fps.sh --all

# 기존 결과를 다시 만들 때만 사용
bash scripts/extract_aria_recording_native_fps.sh --all --overwrite
```

원본 FPS 추출은 FFmpeg rate conversion을 하지 않아 source video frame 하나당 JPEG 하나를 만든다.
5fps 결과와 분리된 `30fps_raw/`에 기록되며, 이미 완성된 시퀀스는 자동으로 건너뛴다.

## 2. 기본 작업 흐름

1. 첫 프레임에서 `n`으로 새 object ID를 만든다.
2. `n`/`h`, `[`/`]`, `g`로 active ID를 먼저 정한 뒤 대상 내부를 좌클릭하거나, 대상을 감싸서 좌드래그한다.
3. 필요하면 대상 밖/원치 않는 부분을 우클릭해 negative point를 추가한다.
4. `Space`로 다음 프레임으로 이동한다. SAM2가 현재 object를 다음 프레임으로 전파한다. `,`로
   이전 프레임을 열 수 있으며, 이때 active ID는 비워진다. `g` 또는 `[`/`]`로 ID를 다시 고른 뒤
   그 프레임을 수정하고 `Space`를 누르면 이후 프레임만 다시 전파한다. `Space`를 누르면
   mouse-prompt와 brush edit mode, active ID는 항상 초기화된다.
5. 구조 변화나 tracking 오류가 보이는 프레임에서 `e`를 누르고 창 안의 입력창에 ID를 입력한다. 이후 영상에서
   좌클릭으로 포함할 점(positive), 우클릭으로 제외할 점(negative)을 찍어 해당 ID의 mask를
   보정한다. `e`를 한 번 더 누르면 이 모드를 끝낸다. 필요할 때만 `b`로 brush 보정을 사용한다.
6. 겹치거나 일시적으로 가려진 frame에서는 해당 ID를 선택한 뒤 `f`를 누른다. 이 frame의 mask만
   annotation에서 제거하며, SAM2 메모리와 다음 frame의 track은 유지된다. `d`와 달리 ID를 종료하지 않는다.
7. `s`로 중간 저장한다. `q` 또는 창 닫기 시에도 자동 저장된다.

화면의 mask 색, 윤곽선, `id:<번호>` 라벨로 현재 추적 상태를 확인한다. 활성 ID에는 `*`가 붙는다.

## 3. 마우스 조작

| 조작 | 동작 |
| --- | --- |
| 좌클릭 | 활성 ID에 positive point 추가 |
| 좌드래그 | 활성 ID에 box prompt 적용 |
| 우클릭 | 활성 ID에 negative point 추가 |
| 편집 모드에서 좌드래그 | mask 영역 추가 |
| 편집 모드에서 우드래그 | mask 영역 제거 |

마우스는 active ID를 만들거나 바꾸지 않으며, 클릭한 mask를 선택하지도 않는다. ID 생성은 `n`/`h`,
선택은 `[`/`]` 또는 `g`로만 한다. box는 active ID의 이전 점 prompt를 대체한다.

## 4. 키보드 단축키

| 키 | 동작 |
| --- | --- |
| `Space`, `.` | 다음 프레임으로 전파 |
| `,` | 이전 프레임으로 이동 (수정 전에는 기존 저장 mask를 표시) |
| `[` / `]` | 이전 / 다음 활성 ID 선택 |
| `n` | 새 object ID 생성 |
| `h`, `Shift+n` | 새 hand ID 생성 |
| `g` | 창 안에 번호를 입력해 ID 선택 또는 새 ID 생성 |
| `d` | 창 안에 번호를 입력해 ID 하나 이상 삭제 |
| `f` | 활성 ID mask를 **현재 frame에서만** 제거; SAM2 메모리·다음 frame은 유지 |
| `c`, `r` | 창 안에 새 번호를 입력해 활성 ID 변경 |
| `x` | 활성 ID의 현재 프레임 point prompt 초기화 |
| `e` | 창 안에 ID 입력 뒤 해당 ID의 **마우스 prompt 편집 모드** 시작/종료 (좌클릭 positive, 우클릭 negative) |
| `p` | 현재 활성 ID의 마우스 prompt 편집 모드 바로 열기 |
| 입력창 | 숫자 입력, `Backspace` 삭제, `Enter` 확정, `Esc` 취소. 터미널 입력은 필요 없음. |
| `v` | 현재 ID·면적·prompt 상태를 터미널에 표시 |
| `b` | brush mask 편집 모드 시작/취소 |
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

`s`를 누르거나 창을 닫아 최종 저장하면 `lineage_graph.png`도 자동 생성/갱신된다. 이 파일은
각 object ID의 mask가 존재하는 프레임 구간과 구조 변화 화살표를 한 장에 그린 검수용 타임라인이다.
초록은 자동 relation, 빨강은 `needs_review`, 주황 `?`는 successor가 아직 없는 종료 ID를 뜻한다.

## 6. 저장 결과

각 입력은 다음 경로에 저장된다.

```text
/workspace/shared/aria_recording/5fps_sampled_masks/<입력_이름>/
├── annotations_ytvis.json
├── interactive_session_meta.json
├── lineage_relations.json
├── lineage_graph.png
├── transparent_crops/<영상_이름>/  # t를 눌렀을 때
└── outline_only/<영상_이름>/        # o를 눌렀을 때
```

- `annotations_ytvis.json`: track ID별 COCO-style uncompressed RLE mask, bbox, area
- `interactive_session_meta.json`: 처리한 프레임, 영상 입력과 결과 파일 위치
- `lineage_relations.json`: 자동 생성된 separation/joining predecessor–successor 관계 및 검수 상태
- `lineage_graph.png`: ID별 mask 타임라인과 relation 화살표를 담은 빠른 검수용 그래프

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
