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

## 3. Implementation Note
In the AttenX project, the DAMSM weights are kept frozen during GAN training to provide a stable evaluation metric and loss signal.
