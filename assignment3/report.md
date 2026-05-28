Problem #1: Learning the New Token
Diffusion personalization refers to the task of teaching a diffusion-based model to cap-
ture a novel subject from only a handful of examples and then express that subject in
diverse images by leveraging the models pre-existing visual knowledge. Textual Inversion
is a representative approach to this diffusion personalization task. By supplying just 3-5
reference images of a new concept, we introduce a placeholder token whose embedding
is fine-tuned while the rest of the model (text encoder, U-Net, VAE) remains frozen. This
targeted update gives precise control over generation via natural language prompts.
For detailed explanations, please refer to the accompanying Jupyter Notebook and the
original paper:
An Image is Worth One Word:
Personalizing Text-to-Image Generation using Textual Inversion
1
In this problem, you will implement the essential steps of Textual Inversion, training a
placeholder token to capture a specific visual concept by optimizing only its embedding
within the pre-trained diffusion framework:
a) Set up the tokenizer and handle the initializer token.
b) Configure the model so that only the necessary components are updated during train-
ing.
c) Complete the interior of the training function.
Problem #2: Use & Evaluate
In this problem, you will experiment with various settings to evaluate how well Textual
Inversion performs under different conditions.
a) Write code that generates images at regular step intervals during the denoising pro-
cess.
b) Observe and describe what each image reveals–what is emphasized at the initial steps,
what appears at the final steps, and any patterns you observe across timesteps.
c) Design and conduct experiments under modified conditions to probe failure cases of
Textual Inversion, exploring variations such as different reference images or prompts,
altered learning rates or numbers of training steps, and comparing outputs before
training (initializer tokens) versus after training (placeholder tokens).
d) Synthesize your qualitative observations and any quantitative measurements to dis-
cuss the advantages and limitations of optimizing only the token embeddings.
e) Justify your conclusions with evidence drawn from your experiments.
Problem #3: Additional Experiments: Success/Failure Anal-
ysis and Model Improvement Proposals
In this problem, you are encouraged to freely experiment with various settings in order
to analyze the behavior of the model, identify limitations, and explore ways to improve
performance. For each experiment, provide both qualitative (e.g., visual inspection) and
quantitative (e.g., CLIP scores) analysis.
a) Conduct an in-depth analysis of when the method succeeds versus fails under the
conditions above, and document your findings.
b) Propose concrete strategies for model improvement (e.g., data augmentation, altered
training procedures, modified loss functions).
2
c) Based on your proposals, design and implement experimental code, and report your
results.

Evaluation reporting guidelines:

- **Quantitative metrics:** report CLIP-based scores per checkpoint and per
	inference configuration: CLIP-I (image-to-reference similarity) and CLIP-T
	(text-to-image alignment). Tabulate results in CSV and include the exact
	prompt/guidance/steps used for each row.
- **Qualitative artifacts:** save representative generated images (per
	checkpoint and per promising inference setting) so visual inspection can be
	correlated with numeric metrics.
- **Failure analysis:** for each observed failure, record the conditions
	(reference bias, prompt mismatch, lr/steps, etc.) and hypothesize root causes.
- **Mitigations & next steps:** propose concrete fixes (data augmentation,
	lower lr, CLIP-based regularizer, additional examples) and recommend which to
	run next based on cost vs. expected benefit.

---
## Answers (Korean)

**Problem 1.a**
우선 `placeholder_token`을 tokenizer에 추가하고 `initializer_token`의 id를 찾아 그 임베딩을 복사하여 placeholder의 초기값으로 설정합니다. 이렇게 하면 학습 초기에 안정적인 시작 임베딩을 얻습니다.

**Problem 1.b**
학습 구성에서 업데이트 대상만 제한하는 구조는 `text_encoder`의 입력 임베딩만 optimizer에 전달하는 방식으로 구현됩니다. 나머지 파라미터(UNet, VAE 등)는 optimizer에 포함시키지 않거나 gradient를 차단하여 임베딩만 갱신되도록 보장합니다.

**Problem 1.c**
훈련 루프의 핵심 흐름은 데이터로더에서 이미지 배치를 읽고 VAE로 latent를 얻은 뒤 노이즈를 샘플링하여 noisy_latents를 만듭니다. text_encoder로 텍스트 임베딩을 얻어 UNet에 넣어 노이즈를 예측하고 MSE 손실을 계산해 placeholder 임베딩에만 역전파합니다. 주기적으로 체크포인트를 저장합니다.

**Problem 2.a**
이미지 생성 단계별 저장은 denoising 루프의 특정 스텝에서 중간 결과를 디코딩하거나, 추론 시 `num_inference_steps`에 따른 최종 이미지를 여러 설정으로 생성해 저장하는 방식입니다. 본 코드에서는 지정된 단계 간격으로 생성 이미지를 저장해 변화 양상을 관찰합니다.

**Problem 2.b**
초기 단계에서는 전체적인 색상·구조 같은 거시적 특징이 형성되고 중간 단계에서 형태와 구성요소가 구체화되며, 최종 단계에서 텍스처·세부선이 더해져 완성됩니다. 단계별 이미지는 후반부로 갈수록 디테일이 점진적으로 등장하는 패턴을 보였습니다.

**Problem 2.c**
실험 설계는 레퍼런스 이미지 변경, prompt 문구 다양화, 학습률 및 학습 스텝 수 조정 등을 포함합니다. 예를 들어 학습률을 과도하게 높이면 임베딩이 불안정해져 CLIP-I가 하락하고, 레퍼런스 다양성이 낮으면 일반화 실패가 관찰됩니다.

**Problem 2.d**
임베딩만 최적화하는 장점은 계산·메모리 비용이 작고 짧은 시간에 개인화가 가능하다는 점입니다. 반면 표현력의 한계로 복잡한 변형이나 프롬프트 문맥에 대한 완전한 적응은 어렵고, 텍스트-이미지 정합성 향상을 위해 추가 튜닝이나 손실항을 도입해야 할 수 있습니다.

**Problem 2.e**
결론은 정성적 관찰(이미지 변화)과 정량적 지표(CLI P-I/T)를 함께 고려해 도출했습니다. 예컨대 특정 lr/step 조합에서 CLIP-T는 상승했지만 CLIP-I가 임계치 미만이면 개념 유지 실패로 해석했습니다. 이러한 수치와 시각적 증거를 근거로 결론을 세웠습니다.

**Problem 3.a**
성공은 레퍼런스 이미지가 다양한 각도·조명으로 구성되어 있고 placeholder 토큰이 프롬프트 문맥에 자연스럽게 삽입될 때 주로 발생했습니다. 실패는 레퍼런스 편향, 배경 차이, 또는 과도한 학습률로 임베딩이 불안정해질 때 주로 관찰되었습니다.

**Problem 3.b**
개선 전략으로는 데이터 증강(다양한 각도·조명), 보수적 학습률과 학습 스케줄링, 체크포인트 기반 조기중단, CLIP-T 정규화 항 같은 텍스트-이미지 정합성 항을 손실에 추가하는 것을 제안합니다.

**Problem 3.c**
구현된 실험은 저장된 `learned_embeds` 체크포인트들을 불러와 기본 추론을 수행해 CLIP-I/CLIP-T를 수집한 뒤, prompt/guidance/steps를 그리드 탐색해 최적 추론 설정을 찾는 방식으로 정량·정성 평가를 수행했습니다.

**Problem 3.d**
종합적으로 임베딩 최적화는 저비용으로 빠른 개인화를 가능하게 하지만 텍스트 반영력 향상은 추론 튜닝이나 재학습을 병행해야 합니다. 따라서 실무에서는 체크포인트 비교, 추론 하이퍼파라미터 탐색, 그리고 필요시 보수적 재학습을 권장합니다.