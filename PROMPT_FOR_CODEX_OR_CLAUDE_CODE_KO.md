# A6000 WSL 설치·실행 요청 프롬프트

아래 내용을 Codex 또는 Claude Code에 그대로 전달하면 된다.

```text
`sam2_annotator` 저장소를 Ubuntu WSL2 + NVIDIA RTX A6000 환경에서 바로 실행 가능하게 설정해줘.

작업 위치:
1. 저장소를 원하는 Linux 경로에 clone한다.
2. 저장소 루트에서 `bash scripts/setup_wsl_a6000.sh`를 실행한다.
   - Conda 환경 이름은 반드시 `sam2-annotator`를 사용한다.
   - CUDA는 A6000 호환 PyTorch cu121 wheel을 사용한다.
   - 스크립트가 Meta 공개 SAM2.1 large checkpoint를 다운로드하고 SHA-256 검증까지 완료해야 한다.
3. `conda run -n sam2-annotator python scripts/verify_a6000.py`가 A6000 GPU와
   `SAM2RealtimePredictor loaded successfully`를 출력하는지 확인한다.
4. WSLg GUI가 가능한 환경이면 `python desktop_annotator.py --pick`을 실행해 OpenCV 창과
   파일 선택창이 뜨는지 확인한다. Windows의 영상은 `/mnt/c/...` 경로로 선택할 수 있다.

제약:
- `data/`, `outputs/`, checkpoint `.pt` 파일은 Git에 추가하거나 삭제하지 않는다.
- `desktop_annotator.py`, `native_ui/`, `vendor/sam2_realtime/`의 동작을 임의로 리팩터링하지 않는다.
- 설치가 실패하면 원인, 실행한 명령, 필요한 최소 수정만 보고하고 진행 전에 확인을 요청한다.
```
