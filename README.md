# Transformer-scratch
### Pytorch Implementation of Transformer Model Presented on ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf)
<img src="imgs/attention-title.PNG" width="500" height="300"></img>

## The Transformer - Model Architecture
- Encoder-Decoder Structure
- Encoder consists of Encoder Block and Decoder consists of Decoder Blocks (in paper: 6 stacks each) <br>
<img src="imgs/transformer-architecture.PNG" width="400" height="500"></img>

## (1) Embedding - Positional Embedding
- Transformer assumes sequential data (seq2seq input-output structure) but Multi-Head Attention inherently lacks information about locality and order.
- Inputs for Encoder (also for Decoder) should first enter Positional Embedding Layer so the input tokens contains information about relative order.
- Note that Positional Embedding (Encoding) vectors are not variables. Same location in the positional embedding vector should always contain same value regardless of samples or batches
- Since Sine function is a periodic function, to avoid different location in a positional embedding vector sharing a same value, Cosine is also employed.
- Positional Embedding Vectors calculated from below formulation are added to inputs of Encoder and Decoder <br>
<img src="imgs/positional-embedding.PNG" width="300" height="50"></img>

## (2) Multi-Head Attention

## (3) Multi-Head Attention

## (4) Encoder Block and Decoder Block

## (5) Encoder

## (6) Decoder


**Mechanism of Multi-Head Attention**

![image](https://user-images.githubusercontent.com/69974410/185332384-fae1ea8f-3f97-4e14-8072-04a19d0176d7.png)

![image](https://user-images.githubusercontent.com/69974410/185332509-f452d2d9-5037-4358-83a9-acfa70357756.png)
