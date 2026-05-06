# DETR Assignment 2 문항별 답변 정리

이 문서는 `CSE4007_Artificial_Intelligence_2026___Assignment_2.pdf`의 각 불렛 문항에 대해, `assignment2_DETR.ipynb` 코드상 위치와 구현 내용을 연결해서 보고서에 어떻게 답변하면 되는지 정리한 자료이다. 설명 수준은 공대생 보고서에 맞춰 **구조, 수식, 텐서 shape, 검증 포인트** 중심으로 작성했다.

---

## PDF 요구사항 요약

제출물은 코드와 보고서를 포함해야 하며, 보고서에는 각 질문에 대한 답변과 결과가 들어가야 한다. PyTorch 사용이 요구되며, PyTorch로 충분히 가능한 연산에 NumPy를 과도하게 쓰면 감점될 수 있다. 또한 직접 구현해야 하는 모델/함수에 대해 sklearn 등 외부 라이브러리의 완성된 기능을 사용하면 0점 처리될 수 있다고 명시되어 있다.

문항 구성은 다음과 같다.

- Problem #1: Building DEtection TRansformer (DETR)
- Problem #2: Building Loss Function for DETR
- Problem #3: Achieving Best Performance

---

# Problem #1: Building DETR

## 1-a. Implement the positional encoding of DETR

### 코드상 위치

- 노트북 위치: `assignment2_DETR.ipynb`
- 주요 cell: **Cell 17**
- 주요 class/function:
  - `DetrSinePositionEmbedding`
  - `DetrSinePositionEmbedding.forward(feature_map, pixel_mask)`
  - `build_position_encoding(d_model)`
  - `DetrConvModel.forward(images, pixel_mask)`에서 backbone feature와 positional embedding을 함께 반환

### 코드에서 어떻게 나타나는가

코드에서는 CNN backbone이 만든 feature map의 각 spatial 위치에 대해 2D sinusoidal position embedding을 생성한다. 입력은 feature map과 pixel mask이고, mask의 누적합을 이용해 x/y 좌표를 만든다.

핵심 구현 흐름은 다음과 같다.

```python
y_embed = not_mask.cumsum(1, dtype=torch.float32)
x_embed = not_mask.cumsum(2, dtype=torch.float32)

dim_t = torch.arange(self.embedding_dim, dtype=torch.float32, device=x.device)
dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode="floor") / self.embedding_dim)

pos_x = x_embed[:, :, :, None] / dim_t
pos_y = y_embed[:, :, :, None] / dim_t
```

그 뒤 짝수 차원에는 `sin`, 홀수 차원에는 `cos`를 적용하고, x/y 방향 embedding을 concat하여 최종적으로 Transformer가 사용할 위치 정보를 만든다.

### 보고서 답변 방향

DETR은 Transformer 구조를 사용하지만 CNN feature map은 이미지의 2D 공간 구조에서 나온다. Transformer의 attention은 기본적으로 순서나 위치를 알지 못하므로, 각 feature가 이미지의 어느 좌표에서 나온 것인지 알려주는 positional encoding이 필요하다.

보고서에는 다음 수식을 넣으면 좋다.

```text
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

본 구현에서는 위 sinusoidal encoding을 x축과 y축에 각각 적용한 뒤 concat하여 2D position embedding으로 확장했다. 최종 embedding은 feature map과 같은 H, W를 가지며, 채널 차원은 `d_model`과 맞아 encoder/decoder attention에서 feature에 더할 수 있다.

### 검증 포인트

- 출력 shape이 `(B, d_model, H, W)`인지 확인한다.
- backbone feature map의 spatial size와 position embedding의 spatial size가 같은지 확인한다.
- encoder 입력으로 flatten했을 때 feature sequence와 position sequence의 shape이 모두 `(B, H*W, d_model)`인지 확인한다.

---

## 1-b. Implement the attention module of DETR

### 코드상 위치

- 노트북 위치: `assignment2_DETR.ipynb`
- 주요 cell: **Cell 19**
- 주요 class/function:
  - `DetrAttention`
  - `DetrAttention.__init__`
  - `DetrAttention.forward(...)`
  - `DetrAttention._shape(...)`
  - `DetrAttention.with_pos_embed(...)`

### 코드에서 어떻게 나타나는가

`DetrAttention.__init__`에서는 query, key, value, output projection을 모두 `nn.Linear`로 정의한다.

```python
self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
```

forward에서는 먼저 position embedding을 hidden state 또는 key/value state에 더한 뒤 Q/K/V를 만든다. 이후 multi-head 형태로 reshape한다.

```python
query_states = self.q_proj(hidden_states) * self.scaling
key_states = self._shape(self.k_proj(key_value_states), source_len, batch_size)
value_states = self._shape(self.v_proj(key_value_states_original), source_len, batch_size)
```

attention score는 scaled dot-product 방식으로 계산된다.

```python
attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
attn_weights = nn.functional.softmax(attn_weights, dim=-1)
attn_output = torch.matmul(attn_weights, value_states)
attn_output = self.out_proj(attn_output)
```

### 보고서 답변 방향

Attention module은 DETR encoder와 decoder의 핵심 구성 요소이다. 수식은 다음과 같다.

```text
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
```

`d_model=256`, `num_heads=8`인 경우 각 head의 차원은 `head_dim=32`가 된다. 입력 `(B, seq_len, 256)`은 attention 계산을 위해 `(B, 8, seq_len, 32)`로 변환된다. 여러 head가 서로 다른 관점에서 feature 간 관계를 학습하고, 마지막에 concat 후 `out_proj`를 통해 다시 `d_model` 차원으로 합쳐진다.

Self-attention과 cross-attention의 차이는 다음처럼 설명하면 된다.

- Encoder self-attention: Q, K, V가 모두 이미지 feature에서 나온다.
- Decoder self-attention: Q, K, V가 모두 object query에서 나온다.
- Decoder cross-attention: Q는 object query, K/V는 encoder output에서 나온다.

### 검증 포인트

- `embed_dim`이 `num_heads`로 나누어떨어지는지 확인한다.
- attention weight shape이 `(B, num_heads, target_len, source_len)`인지 확인한다.
- softmax 후 마지막 차원의 합이 1에 가까운지 확인한다.
- 최종 output shape이 `(B, target_len, d_model)`로 유지되는지 확인한다.

---

## 1-c. Implement the encoder layers of DETR

### 코드상 위치

- 노트북 위치: `assignment2_DETR.ipynb`
- 주요 cell: **Cell 21**
- 주요 class/function:
  - `DetrEncoderLayer`
  - `DetrEncoderLayer.__init__`
  - `DetrEncoderLayer.forward(...)`
  - `DetrEncoder`
  - `DetrEncoder.forward(...)`

### 코드에서 어떻게 나타나는가

`DetrEncoderLayer`는 self-attention, residual connection, layer normalization, feed-forward network로 구성된다.

```python
self.self_attn = DetrAttention(
    embed_dim=d_model,
    num_heads=encoder_attention_heads,
)
self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
self.fc1 = nn.Linear(self.embed_dim, encoder_ffn_dim)
self.fc2 = nn.Linear(encoder_ffn_dim, self.embed_dim)
self.final_layer_norm = nn.LayerNorm(self.embed_dim)
```

forward에서는 먼저 self-attention을 적용한 후 residual connection과 layer normalization을 수행한다.

```python
residual = hidden_states
hidden_states = self.self_attn(
    hidden_states=hidden_states,
    attention_mask=attention_mask,
    object_queries=object_queries,
)
hidden_states = residual + hidden_states
hidden_states = self.self_attn_layer_norm(hidden_states)
```

그 다음 FFN을 적용한다.

```python
residual = hidden_states
hidden_states = self.fc1(hidden_states)
hidden_states = self.activation_fn(hidden_states)
hidden_states = self.fc2(hidden_states)
hidden_states = residual + hidden_states
hidden_states = self.final_layer_norm(hidden_states)
```

`DetrEncoder`는 이러한 encoder layer를 여러 개 쌓아 순차적으로 통과시킨다.

### 보고서 답변 방향

Encoder는 backbone이 추출한 이미지 feature sequence를 입력으로 받아, 이미지 내 모든 위치 간의 관계를 self-attention으로 학습한다. 예를 들어 자동차의 바퀴, 차체, 창문처럼 멀리 떨어져 있어도 같은 객체에 속하는 feature들이 서로 정보를 주고받을 수 있다.

Encoder layer의 구조는 다음과 같이 정리할 수 있다.

```text
Input
 -> Multi-Head Self-Attention
 -> Residual Add + LayerNorm
 -> Feed Forward Network
 -> Residual Add + LayerNorm
 -> Output
```

### 검증 포인트

- encoder layer를 지나도 shape이 `(B, seq_len, d_model)`로 유지되는지 확인한다.
- layer를 여러 번 쌓아도 hidden state shape이 유지되는지 확인한다.
- NaN/Inf가 발생하지 않는지 확인한다.
- attention mask가 padding 영역에 적용되는지 확인한다.

---

## 1-d. Implement the decoder layers of DETR

### 코드상 위치

- 노트북 위치: `assignment2_DETR.ipynb`
- 주요 cell: **Cell 23**
- 주요 class/function:
  - `DetrDecoderLayer`
  - `DetrDecoderLayer.__init__`
  - `DetrDecoderLayer.forward(...)`
  - `DetrDecoder`
  - `DetrDecoder.forward(...)`

### 코드에서 어떻게 나타나는가

`DetrDecoderLayer`는 self-attention, encoder-decoder cross-attention, FFN으로 구성된다.

```python
self.self_attn = DetrAttention(...)
self.encoder_attn = DetrAttention(...)
self.fc1 = nn.Linear(self.embed_dim, decoder_ffn_dim)
self.fc2 = nn.Linear(decoder_ffn_dim, self.embed_dim)
```

첫 번째 단계는 object query 간 self-attention이다.

```python
hidden_states = self.self_attn(
    hidden_states=hidden_states,
    object_queries=query_position_embeddings,
    attention_mask=attention_mask,
)
```

두 번째 단계는 object query가 encoder output을 참조하는 cross-attention이다.

```python
hidden_states = self.encoder_attn(
    hidden_states=hidden_states,
    object_queries=query_position_embeddings,
    key_value_states=encoder_hidden_states,
    attention_mask=encoder_attention_mask,
    spatial_position_embeddings=object_queries,
)
```

마지막으로 FFN을 통과한다.

```python
hidden_states = self.fc1(hidden_states)
hidden_states = self.activation_fn(hidden_states)
hidden_states = self.fc2(hidden_states)
```

### 보고서 답변 방향

Decoder는 `num_queries=100`개의 object query를 입력으로 사용한다. 각 query는 이미지 안에서 하나의 객체 후보를 담당하는 슬롯처럼 동작한다.

Decoder layer는 다음 세 단계로 설명하면 된다.

```text
1. Self-attention: object query끼리 서로 정보를 교환한다.
2. Cross-attention: object query가 encoder의 이미지 feature를 조회한다.
3. FFN: query representation을 비선형 변환한다.
```

Cross-attention이 DETR의 핵심이다. 각 query는 encoder output 전체를 보면서 자신이 담당할 객체의 위치와 종류에 대한 정보를 업데이트한다. 따라서 decoder output은 각 query별 객체 표현이 되고, 이후 prediction head에서 class와 box로 변환된다.

### 검증 포인트

- decoder input/output shape이 `(B, num_queries, d_model)`로 유지되는지 확인한다.
- cross-attention에서 query 길이는 `num_queries`, key/value 길이는 `H*W`가 되는지 확인한다.
- decoder 최종 output이 prediction head에 들어갈 수 있는지 확인한다.

---

## 1-e. Implement the heads and complete architecture of DETR

### 코드상 위치

- 노트북 위치: `assignment2_DETR.ipynb`
- 주요 cell: **Cell 25**
- 주요 class/function:
  - `DetrModel`
  - `DetrModel.forward(images)`
  - `DetrMLPPredictionHead`
  - `DetrForObjectDetection`
  - `DetrForObjectDetection.forward(images)`
  - `DetrForObjectDetection.post_process(...)`

### 코드에서 어떻게 나타나는가

`DetrModel`은 backbone, 1x1 projection, encoder, decoder를 연결한다.

```python
self.input_projection = nn.Conv2d(
    backbone.intermediate_channel_sizes[-1], d_model, kernel_size=1
)
self.query_position_embeddings = nn.Embedding(num_queries, d_model)
self.encoder = DetrEncoder(...)
self.decoder = DetrDecoder(...)
```

forward에서는 backbone feature를 얻고, 마지막 feature map에 1x1 convolution을 적용하여 channel을 `d_model`로 줄인다.

```python
projected_feature_map = self.input_projection(feature_map)
flattened_features = projected_feature_map.flatten(2).permute(0, 2, 1)
object_queries = object_queries_list[-1].flatten(2).permute(0, 2, 1)
flattened_mask = mask.flatten(1)
```

`DetrForObjectDetection`은 DETR base model 위에 class head와 box head를 붙인다.

```python
self.class_labels_classifier = nn.Linear(d_model, num_labels + 1)
self.bbox_predictor = DetrMLPPredictionHead(
    input_dim=d_model,
    hidden_dim=d_model,
    output_dim=4,
    num_layers=3,
)
```

forward에서는 decoder output을 이용해 class logits와 box를 만든다.

```python
sequence_output = outputs[0]
logits = self.class_labels_classifier(sequence_output)
pred_boxes = self.bbox_predictor(sequence_output).sigmoid()
```

### 보고서 답변 방향

전체 DETR 구조는 다음과 같이 설명하면 된다.

```text
Input image
 -> ResNet50 backbone
 -> 1x1 convolution projection
 -> flatten to sequence
 -> Transformer encoder
 -> Transformer decoder with 100 object queries
 -> class head and box head
```

Class head는 각 query를 VOC 20개 class와 no-object class 중 하나로 분류한다. 따라서 output class 수는 `20 + 1 = 21`이다. Box head는 MLP로 각 query의 bounding box를 `(cx, cy, w, h)` 형식으로 예측하고, sigmoid를 통해 normalized coordinate 범위인 0~1로 제한한다.

### 검증 포인트

- `logits` shape이 `(B, 100, 21)`인지 확인한다.
- `pred_boxes` shape이 `(B, 100, 4)`인지 확인한다.
- `pred_boxes` 값이 sigmoid 후 0~1 범위에 있는지 확인한다.
- `post_process` 후 box가 원본 이미지 좌표계의 `(x1, y1, x2, y2)`로 변환되는지 확인한다.

---

## 1-f. Describe the implementation verification process for each module

### 코드상 위치

이 문항은 특정 TODO 하나가 아니라 전체 구현 검증에 대한 보고서 문항이다. 관련 코드는 다음 cell들에 걸쳐 있다.

- Cell 17: positional encoding, backbone wrapper
- Cell 19: attention
- Cell 21: encoder
- Cell 23: decoder
- Cell 25: full DETR architecture and heads
- Cell 33: training loop
- Cell 35-36: mAP evaluation
- Cell 39-40: prediction visualization

### 코드에서 어떻게 나타나는가

학습 loop에서는 model output과 loss가 정상적으로 계산되고 backward가 수행되는지 확인한다.

```python
outputs = model(images)
loss, loss_dict = loss_fn(outputs, annots)

optimizer.zero_grad()
loss.backward()
optimizer.step()
```

평가에서는 model output을 post-process한 뒤 mAP를 계산한다.

```python
outputs = model(images)
processed_outputs = model.post_process(outputs, threshold=CONF_THRESHOLD)
test_map_iou += map_iou(processed_outputs, annots, num_classes=20, iou_thr=IOU_THRESHOLD)
```

시각화에서는 예측 box가 이미지 위에 제대로 나타나는지 확인한다.

### 보고서 답변 방향

보고서에는 다음처럼 쓰면 된다.

```text
각 모듈의 검증은 shape consistency, numerical stability, and functional behavior 관점에서 수행했다.
```

모듈별 검증 내용은 다음과 같이 정리한다.

| 모듈 | 검증 내용 |
|---|---|
| Positional encoding | feature map과 동일한 H/W, `d_model` 채널 확인 |
| Attention | Q/K/V shape, attention weight softmax, output shape 확인 |
| Encoder | layer 반복 후에도 `(B, seq_len, d_model)` 유지 확인 |
| Decoder | query 수 100개 유지, cross-attention source length 확인 |
| Heads | `logits=(B,100,21)`, `pred_boxes=(B,100,4)` 확인 |
| Loss | finite loss, backward 정상 수행 확인 |
| Inference | post-process 후 원본 이미지 좌표계 box 확인 |

---

# Problem #2: Building Loss Function for DETR

## 2-a. Implement the Hungarian matcher

### 코드상 위치

- 노트북 위치: `assignment2_DETR.ipynb`
- 주요 cell: **Cell 28**
- 주요 class/function:
  - `HungarianMatcher`
  - `HungarianMatcher.forward(outputs, targets)`

### 코드에서 어떻게 나타나는가

먼저 prediction을 batch 차원까지 flatten한다.

```python
out_prob = outputs["logits"].flatten(0, 1).softmax(-1)
out_bbox = outputs["pred_boxes"].flatten(0, 1)
```

target label과 target box도 concat한다.

```python
target_ids = torch.cat([v["class_labels"] for v in targets])
target_bbox = torch.cat([v["boxes"] for v in targets])
```

matching cost는 class cost, bbox L1 cost, GIoU cost를 합쳐 만든다.

```python
class_cost = -out_prob[:, target_ids]
bbox_cost = torch.cdist(out_bbox, target_bbox, p=1)
giou_cost = -generalized_box_iou(
    center_to_corners_format(out_bbox),
    center_to_corners_format(target_bbox),
)

cost_matrix = (
    self.class_cost * class_cost
    + self.bbox_cost * bbox_cost
    + self.giou_cost * giou_cost
)
```

마지막으로 Hungarian algorithm으로 최적 matching index를 얻는다.

```python
indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost_matrix.split(sizes, -1))]
```

### 보고서 답변 방향

DETR은 100개의 query를 항상 예측하지만, 실제 객체 수는 이미지마다 다르다. 따라서 예측과 정답을 단순히 순서대로 비교할 수 없고, 예측 query와 ground truth object 사이의 최적 1:1 matching이 필요하다.

matching cost는 다음과 같이 정의된다.

```text
C_ij = λ_cls C_cls(i,j) + λ_bbox ||b_i - b_j||_1 + λ_giou C_giou(i,j)
```

구현에서는 class 확률이 높고, box L1 distance가 작고, GIoU가 높은 prediction-target pair가 낮은 cost를 갖도록 했다.

주의할 점: PDF에서 외부 라이브러리 사용 제한을 강하게 말하고 있다. 현재 노트북은 `scipy.optimize.linear_sum_assignment`를 사용한다. 과제 skeleton이 이 사용을 전제로 제공한 코드라면 문제 없을 수 있지만, 보고서에는 “Hungarian matching solver를 이용해 최종 assignment를 계산했다” 정도로 표현하고, 직접 구현 요구 범위가 cost matrix 구성에 있었다는 식으로 조심스럽게 쓰는 편이 안전하다.

### 검증 포인트

- 각 batch마다 반환되는 prediction index와 target index 길이가 같은지 확인한다.
- target 수가 prediction 수보다 작을 때도 정상 동작하는지 확인한다.
- class/bbox/GIoU cost matrix shape이 `(B*num_queries, total_num_targets)`인지 확인한다.

---

## 2-b. Implement the object loss

### 코드상 위치

- 노트북 위치: `assignment2_DETR.ipynb`
- 주요 cell: **Cell 29**
- 주요 class/function:
  - `DetrLoss.compute_loss_object(...)`

### 코드에서 어떻게 나타나는가

object loss는 prediction 중 no-object가 아닌 class로 분류된 query 수를 세고, ground truth object 수와 비교한다.

```python
card_pred = (logits.argmax(-1) != self.num_classes).sum(1)
object_error = nn.functional.l1_loss(card_pred.float(), target_lengths.float())
```

여기서 `self.num_classes`는 no-object class index로 사용된다. VOC class가 20개이므로 no-object class index는 20이다.

### 보고서 답변 방향

이 항목은 일반적인 gradient-based training loss라기보다 cardinality error에 가깝다. 모델이 실제 객체 수와 비슷한 개수의 non-empty prediction을 내는지 확인하는 지표이다.

```text
object_error = |#predicted_non_empty_boxes - #ground_truth_boxes|
```

DETR은 set prediction 구조이므로 객체 수 예측도 중요하다. 너무 많은 query가 object로 분류되면 false positive가 증가하고, 너무 적으면 missed detection이 증가한다.

### 검증 포인트

- `argmax` 결과가 no-object class가 아닌 query만 세는지 확인한다.
- batch별 target object 수와 비교되는지 확인한다.
- `torch.no_grad()` 아래에서 logging metric처럼 계산되는지 확인한다.

---

## 2-c. Implement the class loss

### 코드상 위치

- 노트북 위치: `assignment2_DETR.ipynb`
- 주요 cell: **Cell 29**
- 주요 class/function:
  - `DetrLoss.compute_loss_labels(...)`

### 코드에서 어떻게 나타나는가

먼저 모든 query의 target class를 no-object로 초기화한다.

```python
target_classes = torch.full(
    source_logits.shape[:2],
    self.num_classes,
    dtype=torch.int64,
    device=source_logits.device,
)
```

Hungarian matching된 query 위치에는 실제 object class label을 넣는다.

```python
target_classes[idx] = target_classes_o
```

그 후 cross entropy loss를 계산한다.

```python
loss_ce = nn.functional.cross_entropy(
    source_logits.transpose(1, 2),
    target_classes,
    self.empty_weight,
)
```

`empty_weight[-1] = eos_coef`로 no-object class의 loss weight를 낮춘다.

### 보고서 답변 방향

DETR의 class loss는 matching된 prediction은 해당 object class로, matching되지 않은 prediction은 no-object class로 학습한다. VOC2007에는 20개 object class가 있으므로 classifier output은 `20 + 1 = 21`개 class이다.

no-object query가 훨씬 많기 때문에 no-object class에 낮은 weight인 `eos_coef=0.1`을 적용했다. 이를 통해 모델이 배경 class에만 치우치는 것을 완화한다.

### 검증 포인트

- `source_logits` shape이 `(B, num_queries, num_classes+1)`인지 확인한다.
- `cross_entropy` 입력을 위해 `transpose(1, 2)` 후 shape이 `(B, num_classes+1, num_queries)`가 되는지 확인한다.
- matching되지 않은 query가 no-object class로 처리되는지 확인한다.

---

## 2-d. Implement the bounding box loss

### 코드상 위치

- 노트북 위치: `assignment2_DETR.ipynb`
- 주요 cell: **Cell 29**
- 주요 class/function:
  - `DetrLoss.compute_loss_boxes(...)`

### 코드에서 어떻게 나타나는가

Hungarian matching된 prediction box와 target box만 선택한다.

```python
idx = self._get_source_permutation_idx(indices)
source_boxes = outputs["pred_boxes"][idx]
target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
```

그 후 L1 loss를 계산한다.

```python
loss_bbox = nn.functional.l1_loss(source_boxes, target_boxes, reduction="none")
losses["loss_bbox"] = loss_bbox.sum() / num_boxes
```

### 보고서 답변 방향

Bounding box loss는 matching된 예측 box와 ground truth box의 좌표 차이를 직접 최소화한다. DETR에서는 box를 normalized center format `(cx, cy, w, h)`로 다룬다.

```text
L_bbox = Σ ||b_pred - b_gt||_1 / N
```

여기서 `N`은 batch 내 target box 수이다. 전체 loss 계산에서는 `bbox_loss_coefficient=5`를 곱해 box regression의 중요도를 높인다.

### 검증 포인트

- matching된 box만 loss에 들어가는지 확인한다.
- box format이 prediction과 target 모두 `(cx, cy, w, h)`인지 확인한다.
- loss가 target box 개수 `num_boxes`로 normalize되는지 확인한다.

---

## 2-e. Implement the GIoU loss

### 코드상 위치

- 노트북 위치: `assignment2_DETR.ipynb`
- 주요 cell:
  - **Cell 27**: box utility functions
  - **Cell 29**: `DetrLoss.compute_loss_boxes(...)`
- 주요 function:
  - `center_to_corners_format(...)`
  - `generalized_box_iou(...)`
  - `DetrLoss.compute_loss_boxes(...)`

### 코드에서 어떻게 나타나는가

GIoU 계산을 위해 center format box를 corner format으로 변환한다.

```python
generalized_box_iou(
    center_to_corners_format(source_boxes),
    center_to_corners_format(target_boxes),
)
```

loss는 diagonal matching pair의 GIoU만 사용한다.

```python
loss_giou = 1 - torch.diag(
    generalized_box_iou(
        center_to_corners_format(source_boxes),
        center_to_corners_format(target_boxes)
    )
)
losses["loss_giou"] = loss_giou.sum() / num_boxes
```

### 보고서 답변 방향

L1 loss는 좌표 차이만 측정하므로 실제 box overlap 품질을 직접 반영하지 못한다. 그래서 DETR은 GIoU loss를 함께 사용한다.

```text
L_giou = Σ (1 - GIoU(b_pred, b_gt)) / N
```

GIoU는 box가 겹치지 않는 경우에도 enclosing box를 고려하여 학습 신호를 줄 수 있다. 따라서 위치가 많이 어긋난 초기 학습 단계에서도 box regression을 안정적으로 유도한다.

### 검증 포인트

- GIoU 계산 전에 box가 `(x1, y1, x2, y2)` corner format으로 변환되는지 확인한다.
- matching pair끼리만 비교하기 위해 `torch.diag`를 사용하는지 확인한다.
- 전체 loss에서 `giou_loss_coefficient=2`가 적용되는지 확인한다.

---

## 2-f. Describe the implementation verification process for each loss module

### 코드상 위치

- Cell 28: Hungarian matcher
- Cell 29: `DetrLoss`
- Cell 33: training loop에서 loss 출력

### 코드에서 어떻게 나타나는가

학습 중 loss dictionary를 출력하여 각 loss term이 정상적으로 계산되는지 확인한다.

```python
message = '[Epoch {}: {:>4d}/{:>4d}] total loss: {:>4f}, cls loss: {:>4f}, box loss: {:>4f}, giou loss: {:>4f}, obj loss: {:>4f}'
```

### 보고서 답변 방향

검증은 다음 항목을 중심으로 설명한다.

| Loss module | 검증 내용 |
---|---|
| Hungarian matcher | prediction-target index 길이 일치, cost matrix shape 확인 |
| Class loss | no-object class 처리 및 `eos_coef` 적용 확인 |
| Object loss | predicted object 수와 target object 수 비교 확인 |
| BBox loss | matching된 pair의 L1 loss만 계산되는지 확인 |
| GIoU loss | corner format 변환 후 diagonal pair loss 계산 확인 |
| Total loss | `loss_ce + 5*loss_bbox + 2*loss_giou` 형태로 합산되는지 확인 |

---

# Problem #3: Achieving Best Performance

## 3-a. Join competition and submit predictions

### 코드상 위치

- 노트북 위치: `assignment2_DETR.ipynb`
- 주요 cell:
  - **Cell 36**: test evaluation
  - **Cell 42**: submission CSV 생성
- 주요 file:
  - `submission.csv`
- 주요 function:
  - `save_submission_from_model(...)`

### 코드에서 어떻게 나타나는가

submission 생성 함수는 test dataloader에 대해 inference를 수행하고, 각 이미지의 예측 결과를 CSV로 저장한다.

```python
outputs = model(images)
processed_outputs = model.post_process(
    outputs,
    threshold=CONF_THRESHOLD_FOR_SUBMISSION,
    target_sizes=orig_sizes,
)
```

CSV 형식은 다음과 같다.

```text
image_id,PredictionString
000001,score label x1 y1 x2 y2 score label x1 y1 x2 y2 ...
```

실제 `submission.csv`의 첫 줄도 다음 형식이다.

```text
image_id,PredictionString
000001,0.932518 11 37.496471 223.326920 143.855164 335.494568 ...
```

### 보고서 답변 방향

Competition submission을 위해 test set에 대해 inference를 수행하고 `submission.csv`를 생성했다. 각 row는 image id와 prediction string으로 구성되며, prediction string은 confidence score, class label, bounding box 좌표를 반복해서 포함한다.

Box 좌표는 post-processing에서 원본 이미지 크기로 복원하고, 이미지 범위를 벗어나지 않도록 clamp했다.

### 검증 포인트

- CSV header가 competition 요구 형식과 일치하는지 확인한다.
- 각 prediction이 `score label x1 y1 x2 y2` 순서인지 확인한다.
- box 좌표가 원본 이미지 width/height 범위 안에 있는지 확인한다.

---

## 3-b. Develop current DETR framework to achieve best performance

### 코드상 위치

- 노트북 위치: `assignment2_DETR.ipynb`
- 주요 cell:
  - **Cell 32**: hyperparameter 설정
  - **Cell 33**: pretrained weight load 및 training
  - **Cell 35-36**: mAP 평가
  - **Cell 42**: submission 생성
- 주요 file:
  - `detr_voc2007.pt`
  - `submission.csv`

### 코드에서 어떻게 나타나는가

전처리 설정은 다음과 같다.

```python
SHORTEST_EDGE = 800
LONGEST_EDGE = 1333
RESCALE_FACTOR = 1 / 255
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
```

모델 설정은 다음과 같다.

```python
NUM_QUERIES = 100
ENCODER_LAYERS = 6
DECODER_LAYERS = 6
D_MODEL = 256
BACKBONE_MODEL_TYPE = 'resnet50'
USE_PRETRAINED_BACKBONE = True
```

loss weight는 DETR 기본 설정과 유사하다.

```python
EOS_COEFFICIENT = 0.1
CLASS_COST = 1
BBOX_COST = 5
GIOU_COST = 2
BBOX_LOSS_COEFFICIENT = 5
GIOU_LOSS_COEFFICIENT = 2
```

학습 설정은 다음과 같다.

```python
NUM_EPOCHS = 50
BATCH_SIZE = 4
optimizer = optim.Adam(model.parameters(), lr=1e-5)
```

또한 HuggingFace의 `facebook/detr-resnet-50` checkpoint에서 backbone/transformer 및 bbox head 일부를 로드한다. VOC class 수가 다르기 때문에 class head는 제외한다.

```python
if k.startswith('class_labels_classifier'):
    continue
```

### 보고서 답변 방향

성능 향상을 위해 다음 전략을 사용했다고 설명하면 된다.

1. ResNet50 pretrained backbone 사용
2. DETR pretrained weight 일부 로드
3. VOC2007 class 수에 맞게 classifier head 재학습
4. ImageNet mean/std normalization 적용
5. box regression에 L1 loss와 GIoU loss를 함께 사용
6. confidence threshold를 사용해 낮은 확률 예측 제거

보고서 문장 예시는 다음과 같다.

```text
성능 향상의 핵심은 사전학습된 DETR/ResNet50 feature를 활용하여 VOC2007처럼 상대적으로 작은 데이터셋에서도 안정적으로 수렴하도록 만든 것이다. Classifier head는 VOC 20개 class에 맞게 새로 학습하고, backbone과 transformer의 표현력은 pretrained weight에서 가져왔다.
```

### 검증 포인트

- pretrained checkpoint 로드 시 class head shape mismatch를 피하기 위해 classifier head를 제외했는지 확인한다.
- `NUM_QUERIES=100`이 pretrained checkpoint와 일치하는지 확인한다.
- mAP 평가 코드가 IoU threshold 0.5 기준으로 동작하는지 확인한다.
- submission threshold가 너무 높거나 낮지 않은지 확인한다.

---

## 3-c. Explain how best performance was achieved so far

### 코드상 위치

- Cell 32: hyperparameter
- Cell 33: training strategy
- Cell 35: AP/mAP 계산 함수
- Cell 36: test mAP 평가
- Cell 40: prediction visualization
- Cell 42: submission 저장

### 코드에서 어떻게 나타나는가

mAP 계산은 class별 AP를 계산한 후 평균을 내는 방식이다.

```python
ap = calculate_ap(tp, fp, n_gt)
aps.append(ap)
return torch.stack(aps).mean().item()
```

test loop에서는 batch별 mAP를 누적한 뒤 평균을 낸다.

```python
test_map_iou += map_iou(
    processed_outputs,
    annots,
    num_classes=20,
    iou_thr=IOU_THRESHOLD,
)
test_map_iou /= num_batches
```

### 보고서 답변 방향

보고서에는 “무엇을 했고, 왜 성능에 도움이 됐는지”를 중심으로 쓴다.

예시 답변:

```text
본 구현에서는 pretrained ResNet50 backbone과 pretrained DETR weight를 활용하여 초기 feature representation을 안정화했다. VOC2007은 COCO보다 class 수가 적기 때문에 classification head는 VOC 20개 class와 no-object class에 맞게 새로 정의하였다. 또한 bbox L1 loss와 GIoU loss를 함께 사용하여 좌표 오차와 실제 overlap 품질을 동시에 최적화했다.

Inference 단계에서는 confidence threshold 0.25를 적용하여 낮은 확률의 prediction을 제거했고, post-processing에서 normalized box를 원본 이미지 좌표계로 변환했다. 마지막으로 test set prediction을 competition submission 형식의 CSV로 저장했다.
```

결과를 쓸 때는 실제 실행 결과의 `Test mAP` 값과 competition score를 반드시 넣는 것이 좋다. 현재 정리 문서에서는 실행 로그가 제공되지 않았으므로, 보고서 작성 시 본인의 실행 결과를 다음 형태로 채우면 된다.

```text
Test mAP@0.5: [실제 출력값]
Competition public score: [제출 후 점수]
```

---

# 보고서에 넣기 좋은 최종 구조

```text
1. Introduction
   - Object detection 문제 정의
   - DETR의 set prediction 관점

2. DETR Architecture Implementation
   2.1 Positional Encoding
   2.2 Multi-Head Attention
   2.3 Encoder
   2.4 Decoder
   2.5 Prediction Heads and Full Architecture

3. Loss Function Implementation
   3.1 Hungarian Matcher
   3.2 Class Loss
   3.3 Object/Cardinality Loss
   3.4 Bounding Box L1 Loss
   3.5 GIoU Loss

4. Verification
   - Module-wise shape check
   - Loss finite value check
   - Forward/backward check
   - Visualization check

5. Performance Improvement
   - Pretrained backbone and DETR weight
   - Hyperparameter setting
   - mAP evaluation
   - Submission generation

6. Conclusion
   - 구현 요약
   - 한계
   - 향후 개선 방향
```

---

# 핵심 체크리스트

- [ ] PDF의 각 불렛이 보고서에 명시적으로 대응되는가?
- [ ] 각 구현 설명에 노트북 cell/class/function 위치를 적었는가?
- [ ] 수식과 코드 흐름이 함께 설명되는가?
- [ ] shape 검증을 포함했는가?
- [ ] `logits=(B,100,21)`, `pred_boxes=(B,100,4)`를 언급했는가?
- [ ] Hungarian matching cost 구성식을 설명했는가?
- [ ] class/object/bbox/GIoU loss를 구분했는가?
- [ ] 실제 `Test mAP`와 competition score를 본인 실행 결과로 채웠는가?
- [ ] `submission.csv` 형식을 설명했는가?

