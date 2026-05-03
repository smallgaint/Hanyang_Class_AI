# TODO 목록 — `assignment2/assignment2_DETR.ipynb`

다음은 노트북에서 자동으로 찾아낸 `TODO` 블록들입니다. 각 항목은 편집 가능한 블록(`# [START]` / `# [END]`)의 간단한 위치·요약과 컨텍스트 스니펫입니다. 실제 수정을 할 때는 노트북 내 해당 TODO 블록 안에서만 작업하세요.

총 발견: 20개(대략)

---

1) 위치: position embedding 구현부
- 요약: `DetrSinePositionEmbedding.forward`에서 forward 완성 필요
- 컨텍스트: "# TODO: complete the forward function" / `# [START]` ... `# [END]`

2) 위치: 어텐션 모듈 초기화
- 요약: `DetrAttention.__init__`에서 `k_proj, v_proj, q_proj, out_proj` 정의 필요
- 컨텍스트: "# TODO: fill the below incomplete code" / `self.k_proj =` 등

3) 위치: 어텐션 레이어 (쿼리/키 위치 임베딩 추가)
- 요약: 어텐션 입력에 position embeddings 더하는 코드 필요(힌트: `with_pos_embed` 사용)
- 컨텍스트: "# TODO: add position embeddings to the hidden states before projecting" / `# [START]`..`# [END]`

4) 위치: key-value position embeddings
- 요약: key/value 상태에 위치 임베딩 추가 필요(힌트: `with_pos_embed` 사용)

5) 위치: 어텐션 가중치 계산부
- 요약: `attn_weights =` 계산(쿼리·키 matmul, scaling, softmax 등) 구현 필요

6) 위치: `DetrEncoderLayer.__init__`
- 요약: `self.self_attn`, layer norm, `fc1`, `fc2`, `final_layer_norm` 등 초기화 코드 완성 필요

7) 위치: `DetrEncoderLayer.forward`
- 요약: residual 설정 및 `self.self_attn(...)` 호출 인자 완성, 이후 FFN flow 완성 필요

8) 위치: `DetrEncoder` 초기 입력 처리
- 요약: 입력 임베딩(예: inputs_embeds 변환, positional 추가 등) 초기화 코드 누락

9) 위치: `DetrEncoder` 루프
- 요약: 각 `encoder_layer` 호출 시 `object_queries`를 추가 인자로 전달하도록 루프 완성 필요

10) 위치: `DetrDecoderLayer.__init__`
- 요약: decoder-layer 내 self/encoder attn 모듈, layer norms 등 초기화 누락

11) 위치: `DetrDecoderLayer.forward` (self-attn / cross-attn)
- 요약: residual 처리, self-attn 호출, cross-attn 블록 및 FFN 완성 필요

12) 위치: `DetrDecoder` 클래스
- 요약: `self.layernorm` 정의 및 decoder 루프의 body 완성, 마지막 `hidden_states` 할당 필요

13) 위치: DETR 모델의 피처 프로젝션
- 요약: backbone 출력 feature_map -> `projected_feature_map =` 1x1 conv 등으로 채널 축소 구현 필요

14) 위치: `DetrMLPPredictionHead` 초기화
- 요약: `self.layers =` 부분에 `nn.ModuleList`로 MLP 레이어 구성 필요

15) 위치: `DetrMLPPredictionHead.forward`
- 요약: 레이어 반복 적용(`x = layer(x)`) 부분 구현 필요

16) 위치: `DetrForObjectDetection.__init__`(heads)
- 요약: `self.class_labels_classifier` 정의 필요(클래스 수 + 1(no-object) 주의), bbox MLP 인자 확인 필요

17) 위치: `DetrForObjectDetection.forward`
- 요약: `sequence_output`, `logits`, `pred_boxes` 추출/계산 부분 구현 필요

18) 기타: 여러 곳의 `# TODO: complete the forward function` 블록
- 요약: forward 단계별 연산(정규화, residual 추가, FFN, layernorm 등) 완성 필요

---

사용자 규칙(다시 확인)
- 절대: PyTorch로 구현(필요 시 사전 허가 없이는 numpy 등 사용 금지)
- 수정 범위: 반드시 TODO 블록( `# [START]` ~ `# [END]`) 안에서만 수정
- 변수명 변경 필요 시 사전 승인 요청
- 주석 최소화(간단 키워드 1~3개)

다음 진행 선택지:
- (A) 각 TODO를 우선순위 순으로 하나씩 채워갈까요? (제가 첫 번째 TODO를 채우는 변경을 제안합니다)
- (B) 먼저 로컬 환경 의존성 목록(`requirements.txt`)과 스모크 테스트 스크립트를 생성할까요?

원하시는 다음 작업을 선택해주세요.

**VOC2007 Dataset 설명 및 로컬 환경 적용 방법**

- **무엇을 하는 코드인가**: 노트북의 `VOC2007Dataset` 클래스는 PASCAL VOC 2007 데이터셋을 로컬 디렉토리(기본: `root_dir/VOCdevkit/VOC2007`)에서 로드합니다. 이미지(`JPEGImages/`), 어노테이션(`Annotations/`), 그리고 `ImageSets/Main/{trainval,test}.txt`의 image id 목록을 사용해 데이터셋 인덱싱과 샘플 로드를 수행합니다. 또한 `VOC_download()` 함수는 원격서버에서 tar 아카이브를 내려받아 풀어 `VOCdevkit` 구조를 만듭니다.

- **주요 동작 흐름(요약)**
	- `VOC_download(root_dir)`: `VOCtrainval_06-Nov-2007.tar`와 `VOCtest_06-Nov-2007.tar`를 다운로드하고 압축을 풉니다. (노트: 노트북은 `wget`을 사용)
	- `VOC2007Dataset.__init__(root_dir, is_train, download, transform)`: `download=True`이면 `VOC_download` 호출. 그 다음 `ImageSets/Main/{trainval|test}.txt`에서 image id들을 읽어 `self.image_ids` 구성.
	- `__getitem__(idx)`: 해당 id의 `JPEGImages/*.jpg`를 PIL로 열고, `Annotations/*.xml`을 파싱해 바운딩박스(`xmin,ymin,xmax,ymax`)와 클래스 인덱스를 수집. `boxes`와 `class_labels`를 numpy 배열로 만들어 `annots` dict로 반환. (transform이 있으면 transform 적용)

- **이 코드가 전체 프로그램에서 하는 역할**
	- 데이터 로딩: 학습/검증/테스트 루프에 들어갈 배치 단위 이미지와 정답(박스, 라벨)을 제공합니다.
	- 전처리 체인(transform)이 연결될 수 있도록 annots 구조를 맞춰 전달합니다.
	- 모델 입력 형식(이미지 텐서, 정규화된 bbox 등)으로 변환하기 전의 원시 데이터 획득 지점입니다.

- **로컬(Windows) 환경에서 바꿔야 할 것들**
	1. 경로 변경: 노트북은 `path2data = '/content/VOC'`로 되어 있어 Colab 가정입니다. 로컬에서는 예를 들어 `path2data = r"C:\Users\hohoh\Documents\Programming\Hanyang\인공지능\assignment2\VOC"`처럼 변경하세요. 해당 셀(상단)에 있는 `path2data`만 수정하면 됩니다 — 파일: [assignment2/assignment2_DETR.ipynb](assignment2/assignment2_DETR.ipynb).
	2. 다운로드 전략: 대용량 원격 다운로드 대신 수동으로 VOC tar를 내려받아 `root_dir` 아래에 `VOCdevkit` 폴더를 배치하는 것을 권장합니다. 자동 다운로드를 쓰려면 노트북의 `VOC_download`가 `wget`을 사용하므로 PowerShell에서 `pip install wget` 후 실행하거나 수동으로 tar를 풀어두세요.
		 - 권장(안정적): 브라우저 또는 `curl`/`wget`으로 다운로드한 뒤 압축을 수동으로 푸세요.
		 - PowerShell 압축 해제 예: `tar -xvf VOCtrainval_06-Nov-2007.tar -C .` (Windows 10+의 tar 사용 가능) 또는 7-Zip 사용.
	3. 다운로드 비활성화: 이미 `VOCdevkit`이 있는 경우 `VOC2007Dataset(..., download=False)`로 생성하세요.
	4. 경로 인코딩/한글 폴더 주의: 프로젝트 경로에 한글·공백이 있어도 동작하지만 문제가 생기면 ASCII 경로(예: 사용자 폴더 내 `VOC`)를 쓰는 것이 안전합니다.

- **간단 확인/테스트 스크립트 (노트북 또는 Python 스크립트에서 실행)**
	- PowerShell: 디렉토리 확인
	```powershell
	# 경로 존재 확인
	Test-Path "C:\Users\hohoh\Documents\Programming\Hanyang\인공지능\assignment2\VOC\VOCdevkit\VOC2007\JPEGImages"
	Get-ChildItem -Name "C:\path\to\VOCdevkit\VOC2007\ImageSets\Main\*"
	```

	- Python(노트북 셀에 붙여넣기):
	```python
	path2data = r"C:\Users\hohoh\Documents\Programming\Hanyang\인공지능\assignment2\VOC"
	dataset_train = VOC2007Dataset(root_dir=path2data, is_train=True, download=False)
	print('num samples:', len(dataset_train))
	image, annots = dataset_train[0]
	print(type(image), annots.keys())
	print('boxes shape:', annots['boxes'].shape, 'labels shape:', annots['class_labels'].shape)
	```

- **numpy 사용 관련 주의**
	- 현재 `VOC2007Dataset`는 어노테이션을 `numpy` 배열로 만들고, 전처리 코드(Resize/Normalize/pad 등)도 `numpy` 기반입니다. 사용자 규칙(일관된 PyTorch 우선 사용)을 존중하려면 이 부분을 PyTorch tensor로 바꾸는 리팩터링이 필요합니다. 이 변경은 TODO 블록 외부(데이터 파이프라인 전반)에 영향을 줄 가능성이 크므로 진행 전에 허가를 요청드립니다.

- **권장 작업 순서 (로컬 전환용)**
	1. 로컬 `path2data` 변수만 상단에서 수정.
	2. `VOCdevkit` 폴더가 없다면 수동으로 tar를 내려받아 `root_dir`에 압축 해제.
	3. `dataset_train = VOC2007Dataset(..., download=False)`로 실행해 샘플을 로드해보기.
	4. 문제가 없으면 collate/transform 및 모델 입력 파이프라인으로 이어가는 테스트(스모크 테스트)를 실행.

---
추가로 원하시면 지금 `path2data`를 로컬 경로로 바꾸고(노트북의 해당 셀만 변경), `VOCdevkit` 유무 검사 및 샘플 로드를 제가 대신 실행해 볼게요. 또한 `numpy` 제거 리팩터링을 원하시면 범위·영향(어디를 수정해야 하는지)을 정리해 드리겠습니다.