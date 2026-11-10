# YOLO 객체 탐지 (선택 과제 A)

YOLO11n으로 이미지에서 객체를 탐지하고, 결과를 파일로 저장한 뒤 원본과 대조한 기록을 남긴 프로젝트입니다.

## 결과

| 원본 | 탐지 결과 | 탐지 수 |
|------|----------|--------|
| `samples/bus.jpg` | `outputs/bus_annotated.jpg` | 5건 (bus 1, person 4) |
| `samples/zidane.jpg` | `outputs/zidane_annotated.jpg` | 3건 (person 2, tie 1) |

- 탐지 내역(클래스·신뢰도·박스 좌표·확인 필요 여부): [`outputs/detections.csv`](outputs/detections.csv)
- 실행 요약(모델·장치·탐지 0건 여부): [`outputs/run_log.json`](outputs/run_log.json)
- **원본 대조 기록: [`REVIEW.md`](REVIEW.md)** — 탐지별 판정, 수정 내역, 확인 필요 항목

## 실행 방법

필요한 것: [uv](https://docs.astral.sh/uv/), NVIDIA GPU(없으면 CPU로 동작하지만 느림)

```bash
git clone <이 저장소 주소>
cd yolo-detection
uv sync                                       # 가상환경 + 패키지 설치 (PyTorch 약 2.7GB)
uv run detect.py --source samples/bus.jpg     # 이미지 탐지
uv run detect.py --source samples/zidane.jpg
uv run detect.py --source 0                   # 웹캠 (s: 프레임 저장, q 또는 ESC: 종료) — 미검증, 아래 '한계' 참고
```

- 결과는 `outputs/`에 저장됩니다. 같은 폴더에 다시 실행하면 `detections.csv`와 `run_log.json`에 **이어서 추가**되므로, 처음부터 재현하려면 `outputs/`를 지우고 실행하세요.
- 첫 실행 때 모델 가중치 `yolo11n.pt`(약 5.4MB)를 자동으로 내려받습니다. 가중치 파일은 저장소에 포함하지 않았습니다.
- 옵션: `--conf`(최소 신뢰도, 기본 0.25), `--model`(가중치), `--out`(저장 폴더)

## 실행 환경과 위치

| 항목 | 내용 |
|------|------|
| 실행 위치 | 내 노트북(로컬), Windows 11 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU (CUDA 사용 확인됨) |
| Python | 3.13.15 |
| 패키지 | torch 2.11.0+cu128, torchvision 0.26.0+cu128, ultralytics 8.4.208, opencv-python 5.0.0.93 |
| 도구 | uv 0.13.5 (`uv.lock`으로 버전 고정) |
| 모델 | YOLO11n (COCO 80개 클래스로 학습된 사전학습 가중치, 추가 학습 없음) |

## 도구 선택 이유

- **YOLO11n**: 가장 작은 모델(약 5.4MB)이라 노트북 GPU에서도 부담이 적을 것으로 보고(실시간 속도는 측정하지 않음), 사전학습 가중치를 그대로 써서 별도 학습 없이 탐지를 확인할 수 있어서 선택했습니다. 대신 큰 모델보다 정확도가 낮을 수 있습니다.
- **PyTorch CUDA 12.8 빌드(`cu128`)**: RTX 50 시리즈 GPU는 CUDA 12.8 이상 빌드가 필요해서 `pyproject.toml`에 해당 인덱스를 지정했습니다. 이 설정으로 `torch.cuda.is_available()`이 `True`임을 확인했습니다.
- **uv**: `uv.lock`으로 패키지 버전을 고정해 다른 환경에서도 같은 구성을 재현하기 쉽고, `uv sync` 한 번으로 환경이 만들어집니다. 처음에는 pip로 설치를 시도했고(PyTorch 다운로드가 한 번 멈췄고 아래 경로 문제도 겪음), 이후 재현성을 위해 uv로 전환했습니다. 아래 경로 문제는 uv로 바꿔서 해결된 것이 아니라 짧은 경로로 옮겨서 해결했습니다.
- **신뢰도 0.50 미만은 "확인 필요"로 표시**: 모델이 낮은 확신으로 낸 결과를 확정 결과처럼 보이지 않게 하려는 기준입니다. 0.50은 임의로 정한 값이며 정확도를 보장하는 임계값은 아닙니다.

## 겪은 문제

- Windows 경로 길이 제한(260자): 임시 작업 폴더가 너무 길어 `import torch`가 DLL을 못 불러왔습니다. 프로젝트를 `C:\Users\<사용자>\yolo-detection`처럼 **짧은 경로**에 두어 해결했습니다. 긴 경로에 클론할 때도 같은 이유로 실패할 수 있습니다.

## 한계와 미확인 사항

- **웹캠 실시간 탐지는 검증하지 못했습니다.** `--source 0` 코드는 작성했고 창도 열렸지만, 이 PC에서 카메라 0번이 보내는 프레임의 평균 밝기가 0.0(완전히 검은 영상)이어서 탐지 결과를 얻지 못했습니다(DSHOW는 검은 프레임, MSMF는 프레임 수신 실패). 원인은 카메라 접근 차단 등 PC 쪽 문제로 추정하지만 **확정하지 못했습니다.** 따라서 웹캠 모드의 동작은 이 저장소의 결과로 증명되지 않으며, 제출 결과는 이미지 입력 기준입니다.
- 실행은 **Windows 11 + NVIDIA GPU 한 대에서만** 확인했습니다. macOS/Linux, CPU 전용 환경에서는 `pyproject.toml`의 `cu128` 인덱스 설정 때문에 `uv sync`가 동작하는지 확인하지 못했습니다.
- 검증 이미지는 **샘플 2장**뿐이고, 판정은 사람 눈 확인입니다. 정답 라벨과 비교한 정확도(mAP 등)는 측정하지 않았습니다.
- `zidane.jpg`의 `tie` 탐지(0.45)는 확인 필요로 남겨 두었고, 오른쪽 인물의 넥타이로 보이는 부분이 탐지되지 않은 것은 **미탐지인지 확정하지 못했습니다.** 자세한 내용은 [`REVIEW.md`](REVIEW.md)에 있습니다.
- 탐지 속도(FPS)는 측정하지 않았으므로 성능 수치를 제시하지 않습니다.
- 샘플 이미지는 ultralytics 저장소에 포함된 `bus.jpg`, `zidane.jpg`이며 해당 프로젝트의 라이선스(AGPL-3.0)를 따릅니다.

## 폴더 구조

```
yolo-detection/
├── detect.py          # 탐지 스크립트 (이미지/웹캠)
├── pyproject.toml     # 의존성 (uv)
├── uv.lock            # 버전 고정
├── samples/           # 입력 이미지
├── outputs/           # 결과 이미지, detections.csv, run_log.json
├── REVIEW.md          # 원본 대조 기록
└── README.md
```
