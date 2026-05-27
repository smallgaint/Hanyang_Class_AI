# Textual Inversion CLIP-T 개선 가이드

## 1) 문제 상황 요약
- 현재 관측값 예시: CLIP-I = 0.88, CLIP-T = 0.23
- 해석:
  - CLIP-I가 높다: 학습한 대상의 외형/정체성은 잘 보존됨
  - CLIP-T가 낮다: 프롬프트(문장) 의미 반영력이 상대적으로 약함

즉, "대상은 비슷하게 그리는데 문장 조건을 덜 따르는" 상태입니다.

---

## 2) 이번 개선에서 사용한 방법(무엇을, 어떻게)

### A. 추론 단계 자동 탐색(재학습 없이 즉시 적용)
Cell 37(Additional Experiments)에 다음 아이디어를 넣어 CLIP-T를 올리도록 구성했습니다.

- 탐색 대상
  - prompt 문구 후보
  - guidance_scale 후보
  - num_inference_steps 후보
  - negative_prompt 후보
- 평가 방식
  - 각 조합으로 이미지를 생성
  - CLIP-I, CLIP-T를 각각 계산
  - CLIP-I 하한(예: 0.80) 이상만 통과
  - 통과한 결과 중 CLIP-T 최대값을 최종 선택
- 산출물
  - 최고 설정 1개 출력
  - 전체 탐색 결과 CSV 저장
  - 최고 점수 이미지를 파일로 저장

핵심 포인트는 "CLIP-T 최적화 + CLIP-I 안전장치"를 동시에 거는 것입니다.

### B. 사용한 구성 요소
- 생성 모델: StableDiffusionPipeline
- 스케줄러: DPMSolverMultistepScheduler
- 점수 계산: CLIPModel + CLIPProcessor
- 제약 조건: clip_i_floor (정체성 훼손 방지)

---

## 3) 코드 구조(흐름도)

```text
설정값 정의
  ├─ prompt_candidates
  ├─ guidance_candidates
  ├─ inference_steps_candidates
  └─ negative_prompt_candidates
        ↓
파이프라인/임베딩 로드
  ├─ learned_embeds-step-XXXX.bin
  └─ reference image 선택
        ↓
조합 탐색 루프
  ├─ 이미지 생성
  ├─ CLIP-I 계산 (이미지-이미지)
  └─ CLIP-T 계산 (텍스트-이미지)
        ↓
필터링
  └─ CLIP-I >= clip_i_floor 인 샘플만 유지
        ↓
정렬/선택
  └─ 남은 샘플 중 CLIP-T 최고값 선택
        ↓
결과 저장
  ├─ best setting 출력
  ├─ best image 저장
  └─ search_results.csv 저장
```

---

## 4) 하이퍼파라미터는 어떻게 바꾸면 좋은가

아래는 CLIP-T 개선을 목표로 한 권장 우선순위입니다.

### 4-1. 추론(inference) 하이퍼파라미터
재학습 없이 빠르게 효과를 보는 구간입니다.

1. guidance_scale 탐색
- 권장 범위: 7.5 ~ 13.0
- 일반 경향: guidance를 올리면 텍스트 정합(CLIP-T)이 오르는 경우가 많음
- 주의: 너무 높으면 이미지가 부자연스럽거나 과도한 왜곡 가능

2. num_inference_steps 탐색
- 권장 범위: 30 / 40 / 50
- 일반 경향: 적당히 증가하면 텍스트 반영 안정화에 도움
- 주의: 계산 시간 증가

3. prompt 문구 개선
- "high-quality", "DSLR", "natural light", "close-up" 같은 맥락 키워드 추가
- 대상 토큰(<cat2>)은 유지하되 문장 구조를 다양화

4. negative_prompt 사용
- 예: "blurry, low quality, distorted"
- 목적: 생성 노이즈를 줄여 텍스트 반영을 선명하게 보조

### 4-2. 학습(training) 하이퍼파라미터
CLIP-T를 근본적으로 올리고 싶을 때 적용합니다.

1. learning_rate
- 현재가 5e-4라면 2e-4 ~ 3e-4로 낮춰 비교 권장
- 이유: 텍스트 임베딩이 과하게 흔들리는 것을 완화

2. max_train_steps
- 1000 고정보다 1200~1800까지 확장 실험 권장
- step별 체크포인트(예: 800/1000/1200/1500)에서 CLIP-T 비교 후 최적 step 선택

3. train_batch_size / gradient_accumulation_steps
- 유효 배치 증가가 가능하면(메모리 허용 시) 학습 안정성 개선
- 예: batch 4 유지 + grad accumulation 2

4. prompt template 다양성
- object template만 반복하지 말고 배경/구도/조명 다양화
- 목적: 토큰이 특정 문장 패턴에 과적합되지 않게 함

5. 데이터 구성
- 4~6장 최소 기준보다 8~15장(가능 시)으로 확장
- 서로 다른 각도, 배경, 거리 샷을 섞어 일반화 강화

---

## 5) 추천 실험 프리셋

### Preset A (빠른 개선, 재학습 없음)
- guidance_scale: [7.5, 9.0, 11.0, 13.0]
- num_inference_steps: [30, 40, 50]
- negative_prompt: ["", "blurry, low quality, distorted, extra limbs"]
- 선택 기준: CLIP-I >= 0.80 중 CLIP-T 최대

### Preset B (재학습 포함, 보수적)
- learning_rate: 2e-4
- max_train_steps: 1500
- save_steps: 100
- train_batch_size: 4
- gradient_accumulation_steps: 2
- 선택 기준: step별 CLIP-I/CLIP-T 평균으로 최적 checkpoint 선택

### Preset C (재학습 포함, 공격적)
- learning_rate: 3e-4
- max_train_steps: 1800
- save_steps: 100
- train_batch_size: 4
- gradient_accumulation_steps: 1
- 선택 기준: CLIP-T 우선, 단 CLIP-I 0.80 미만이면 탈락

---

## 6) 결과 해석 기준(실무형)
- 목표는 절대값 하나가 아니라 "동시에" 개선되는 것:
  - CLIP-T 상승
  - CLIP-I는 임계치 이상 유지
- 권장 판정 예시:
  - Good: CLIP-I >= 0.80 and CLIP-T >= 0.28
  - Great: CLIP-I >= 0.85 and CLIP-T >= 0.30

---

## 7) 다음 액션 체크리스트
- [ ] Cell 37 실행 후 best setting 확인
- [ ] search_results.csv에서 상위 5개 시각 검토
- [ ] 가장 자연스러운 2개 설정 고정
- [ ] 필요 시 Preset B로 재학습 후 동일 평가 반복
