# SAM2 Realtime Desktop Annotator

Python/OpenCV 기반 SAM2 비디오 마스크 어노테이터다. 브라우저에서 원본 frame과 mask를
직렬화·합성하지 않으므로, NVIDIA RTX A6000/WSL에서 point·box prompt와 다음 frame 전파를
직접 GPU로 처리한다. `vendor/sam2_realtime`의 동적-ID predictor를 사용해 추적 중에도
object ID를 생성, 삭제, 변경할 수 있다.

## A6000 WSL 빠른 시작

Ubuntu WSL2에서 NVIDIA Windows WSL 드라이버와 Miniconda만 준비된 상태를 기준으로 한다.
WSLg가 활성화돼 있으면 OpenCV 창과 Linux 파일 탐색기가 바로 열린다.

```bash
git clone https://github.com/sys7498/sam2_annotator.git
cd sam2_annotator
bash scripts/setup_wsl_a6000.sh

conda activate sam2-annotator
python desktop_annotator.py --pick
```

설치 스크립트는 A6000 호환 PyTorch 2.5.1/cu121, Python 의존성, OpenCV 런타임 라이브러리와
Meta 공개 SAM2.1 small checkpoint(약 176MB)를 설치하고 GPU 모델 로딩을 검증한다. small은
속도와 video-mask 품질의 기본 균형 모델이며, 필요하면 실행 시 `--checkpoint`와 `--config`로
large 등 다른 SAM2.1 모델을 지정할 수 있다. 모델
checkpoint와 작업 영상/결과는 Git에 포함하지 않는다.

Codex 또는 Claude Code에 환경 구성을 맡길 때는
[PROMPT_FOR_CODEX_OR_CLAUDE_CODE_KO.md](PROMPT_FOR_CODEX_OR_CLAUDE_CODE_KO.md)를 전달하면 된다.

전체 조작법, 모든 단축키, 자동 separation relation과 WSL 문제 해결은
[USAGE_KO.md](USAGE_KO.md)에 정리했다.

Aria raw recordings를 사람별 5fps JPEG 프레임 시퀀스로 만들려면 다음을 사용한다.

```bash
bash scripts/sample_aria_recording_5fps.sh <사람_폴더명>
```

## 실행

```bash
# OS 파일 탐색기로 영상 하나 선택
python desktop_annotator.py --pick

# 경로를 바로 지정
python desktop_annotator.py --input /path/to/video.mp4

# 폴더의 영상 목록에서 TODO/DONE을 보고 선택
python desktop_annotator.py --dataset /path/to/videos
```

결과는 `outputs/<video_name>/annotations_ytvis.json`,
`interactive_session_meta.json`, `lineage_relations.json`, `lineage_graph.png`에 저장된다.
`lineage_graph.png`는 ID별 mask 구간과 separation/joining 화살표를 한눈에 보여 주며, `s`로
중간 저장하거나 창을 닫고 `q`로 종료할 때마다 갱신된다. Windows 쪽 영상은 WSL에서 `/mnt/c/...` 경로로
지정하거나 `--pick`으로 선택할 수 있다.

긴 clip에서 GPU memory를 제한해야 하면 `--state-window 96`을 사용한다. 이 값은 과거
frame 상태를 정리하므로 과거 frame에서 수정하면 해당 frame까지 다시 재생해야 한다.

## 작업 흐름

1. `--pick` 또는 `--input`으로 영상을 연다. 여러 영상은 `--dataset`을 쓰면 목록에서
   완료 파일이 **DONE**, 나머지가 **TODO**로 표시된다.
2. 왼쪽 클릭은 positive point이고, 드래그는 box prompt다. 오른쪽 클릭은 negative point다.
   `Shift`를 함께 누르면 hand ID를 대상으로 한다.
3. `n`은 새 object ID, `h`는 새 hand ID, `[`/`]`는 ID 선택, `d`는 삭제, `c`는 ID 변경이다.
   한 ID를 종료하고 두 ID를 만들면 `1→2 separation`, 두 ID를 종료하고 한 ID를 만들면
   `2→1 joining` predecessor–successor 관계가 자동 생성된다. `l`로 관계를 터미널에
   확인할 수 있고, `lineage_relations.json`과 `lineage_graph.png`에 저장된다. 그래프에서 초록은
   자동 relation, 빨강은 재검수 필요 relation, 주황 `?`는 successor가 아직 없는 종료 ID다. 가림/추적 실패는 관계로 오인하지
   않도록, ID를 명시적으로 종료한 경우에만 자동 생성한다.
4. `Space` 또는 `.`은 한 frame 전진하며 SAM2가 현재 ID를 다음 frame으로 전파한다. `,`는
   이전 frame으로 이동한다. 과거 frame에서 mask를 수정한 뒤 `Space`를 누르면 그 지점 이후만
   앞으로 다시 전파한다.
   `e`는 수정할 ID만 입력한 뒤 영상 위에서 직접 mask를 보정하는 모드다. 이 모드에서
   좌클릭은 positive, 우클릭은 negative prompt이며 `e`를 한 번 더 누르면 종료한다. `p`는
   좌표를 직접 입력해야 할 때 쓰는 터미널 prompt 편집기다. `b`는 brush 보정, `a`는 brush
   보정 mask 적용, `s`는 저장이다.

출력 JSON은 COCO-style uncompressed RLE를 포함한다. 연구 데이터의 target annotation
rate(5 fps)에 맞춰 제출 전 sampling pass를 수행한다.

## 배포 구성

- `native_ui/`: 프로젝트에 포함된 native OpenCV UI
- `vendor/sam2_realtime/`: Apache-2.0 SAM2 realtime source (checkpoint 제외)
- `scripts/setup_wsl_a6000.sh`: A6000 WSL 단일 설치 명령
- `scripts/verify_a6000.py`: CUDA/OpenCV/SAM2 모델 로드 확인

## 경계

이 도구는 mask pass를 위한 도구다. `joining`/`separation`의 timing과
predecessor/successor relation은 독립 temporal pass와 relation pass에서 확정해야
한다. 구조 변화 뒤에는 predecessor ID를 종료하고 successor에 새 ID를 만들어야 한다.
