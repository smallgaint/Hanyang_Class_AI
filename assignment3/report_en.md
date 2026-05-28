Problem #1: Learning the New Token

Diffusion personalization refers to the task of teaching a diffusion-based model to capture a novel subject from only a handful of examples and then express that subject in diverse images by leveraging the model's pre-existing visual knowledge. Textual Inversion is a representative approach to this diffusion personalization task. By supplying just 3-5 reference images of a new concept, we introduce a placeholder token whose embedding is fine-tuned while the rest of the model (text encoder, U-Net, VAE) remains frozen.

## Answers (English)

**Problem 1.a**
Tokenizer initialization and handling of the initializer token: add the `placeholder_token` to the tokenizer, find the `initializer_token` id, and copy its embedding to initialize the placeholder embedding. This provides a stable starting point for training.

**Problem 1.b**
To restrict updates during training, only pass the input embeddings of the `text_encoder` to the optimizer. Keep other model parameters (UNet, VAE, etc.) excluded or set `requires_grad=False`, ensuring that only the token embedding is updated.

**Problem 1.c**
Training loop core flow: read image batches from the dataloader, encode to latents via the VAE, add noise to create noisy latents, encode text with the text encoder, run the UNet to predict noise, compute MSE loss, and backpropagate to update only the placeholder embedding. Save checkpoints periodically.

**Problem 2.a**
Conclusions are drawn from both qualitative observations (image evolution) and
quantitative metrics (CLIP-I/CLIP-T). For reproducible reporting, include a CSV
of metric results and the exact inference parameters used for each run.

**Evaluation reporting guidelines (English)**

- Quantitative metrics: report CLIP-I and CLIP-T per checkpoint and per
	inference configuration; include the prompt, guidance scale, negative prompt,
	and number of inference steps for each measured row.
- Qualitative artifacts: save representative generated images for each
	promising configuration and link them to the CSV rows for visual inspection.
- Failure analysis: document conditions that cause failures (reference bias,
	background mismatch, too-high lr, etc.) and hypothesize why they occur.
- Mitigations & next steps: propose concrete fixes (data augmentation, lower
	lr, CLIP-based regularizer, more examples) and recommend which experiments to
	run next based on estimated cost and potential benefit.

**Problem 2.c**
Failure analysis experiments include changing reference images, varying prompts, and adjusting learning rate or number of training steps. For example, too high a learning rate can destabilize the embedding and reduce CLIP-I, while low diversity in references leads to poor generalization.

**Problem 2.d**
Optimizing only the embedding is computationally efficient and enables quick personalization. However, it limits representational capacity for complex variations and may not fully capture text-conditioned nuances; extra tuning or additional loss terms may be needed to improve text-image alignment.

**Problem 2.e**
Conclusions are drawn from both qualitative observations (image evolution) and quantitative metrics (CLIP-I/CLIP-T). For instance, if CLIP-T improves but CLIP-I drops below a threshold, we interpret this as a failure to retain the concept. These numeric and visual evidences support the conclusions.

**Problem 3.a**
Success occurs when reference images cover varied viewpoints and lighting and when the placeholder token is used naturally in prompts. Failures arise from biased references, background mismatches, or unstable training due to too high learning rates.

**Problem 3.b**
Improvement strategies: data augmentation (views, lighting), conservative learning rates and schedules, checkpoint-based early stopping, and adding a CLIP-T regularization term to the loss to increase text-image alignment.

**Problem 3.c**
Implemented experiments: load saved `learned_embeds` checkpoints, run base inference to collect CLIP-I/CLIP-T, and perform a grid search over prompt/guidance/steps to find optimal inference settings for quantitative and qualitative evaluation.

**Problem 3.d**
Overall, embedding optimization enables low-cost, fast personalization, but improving text responsiveness often requires inference tuning or cautious retraining. In practice, compare checkpoints, search inference hyperparameters, and apply retraining if necessary.
