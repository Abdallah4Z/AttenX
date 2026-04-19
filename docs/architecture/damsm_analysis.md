# Issue #1: Deep Dive - AttnGAN Text-Encoder and DAMSM

## 1. Text Encoder (RNN)
- **Architecture:** Bi-directional LSTM.
- **Input:** Tokenized words from the caption.
- **Output:** 
  - **Global Sentence Embedding:** A fixed-length vector representing the entire caption.
  - **Word-level Embeddings:** A sequence of vectors representing each word in the caption.

## 2. Deep Attentional Multimodal Similarity Model (DAMSM)
The DAMSM is a pre-trained sub-network used to compute the similarity between generated images and the input text descriptions.

### Sub-components:
1. **Inception-v3 Image Encoder:** Encodes the image into local feature maps and a global feature vector.
2. **Text Encoder:** Encodes the text into word and sentence embeddings.

### Loss Function ($L_{DAMSM}$):
- **Word-level Similarity:** Uses an attention mechanism to find relevant words for specific image regions.
- **Sentence-level Similarity:** Computes the cosine similarity between the global sentence embedding and the global image feature.
- **Purpose:** Provides a fine-grained reward signal to the generator, ensuring that specific objects (e.g., "red beak") are rendered correctly.

## 3. Implementation Notes

- DAMSM weights (text and image encoders) are loaded from pre-trained checkpoints and kept frozen during GAN training for stability.
- The generator loss is:
  $$
  L_G = L_{GAN} + \gamma_{damsm} (L_{Words} + L_{Sent})
  $$
  where $\gamma_{damsm}$ is configurable via CLI or environment variable.
- Word-level similarity is masked by caption length to avoid padding artifacts.
- All loss components are logged separately for analysis: word-level, sentence-level, combined DAMSM, and GAN losses.
