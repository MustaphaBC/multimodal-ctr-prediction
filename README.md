Multimodal CTR Prediction (MM-CTR)

This project implements a two-stage multimodal Click-Through Rate (CTR) prediction system combining offline multimodal embedding learning and online CTR prediction with user behavior modeling.

The architecture follows the diagram below:

Task 1 (Offline): Multimodal Embedding via knowledge distillation from a frozen CLIP ViT teacher to a lightweight student encoder.

Task 2 (Online): CTR prediction using a DIN-style attention mechanism over user behavior history.

1. Project Overview
Objective

Predict the probability that a user will click on a target item by jointly modeling:

Visual representation of the target item

Sequential user behavior history

Attention-based interest matching (DIN)

This design is aligned with the MM-CTR Challenge (WWW 2025 – EReL) setting, emphasizing efficiency, modularity, and multimodal learning.

2. System Architecture
Task 1: Multimodal Embedding (Offline)

Goal: Learn compact, high-quality visual embeddings for items.

Pipeline:

Input target item image

Frozen CLIP ViT extracts high-dimensional visual features (teacher)

Student Encoder learns via distillation

Output: 128-dimensional embedding per item

Key Characteristics:

CLIP parameters are frozen

Student encoder is lightweight and deployment-friendly

Embeddings are precomputed and stored

Task 2: CTR Prediction (Online)

Goal: Predict CTR in real time using learned embeddings.

Pipeline:

User behavior sequence (past clicked/viewed items)

Target item embedding as attention query

DIN Attention Layer computes interest relevance

Weighted sum of historical embeddings

MLP predicts CTR score

Advantages:

Captures dynamic user interests

Efficient inference (offline embeddings + lightweight attention)

3. Installation
3.1 Environment Setup
python >= 3.9

Create and activate a virtual environment:

python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\\Scripts\\activate     # Windows

Install dependencies:

pip install -r requirements.txt
4. Training Workflow
Step 1: Offline Embedding Training
python training/train_embedding.py

Outputs:

Student encoder weights

Precomputed 128-d embeddings

Step 2: CTR Model Training
python training/train_ctr.py

Inputs:

User behavior sequences

Precomputed item embeddings

Outputs:

Trained DIN + MLP CTR model

5. Inference
python inference/predict_ctr.py \
  --user_id 123 \
  --item_id 456

Returns:

CTR probability score

6. Evaluation Metrics

AUC (Area Under ROC Curve)

LogLoss

CTR calibration error

7. Key Design Choices

Knowledge Distillation: Reduces inference cost while preserving CLIP semantic power

DIN Attention: Personalized interest modeling

Offline–Online Separation: Improves scalability and latency

8. Future Improvements

Multimodal fusion (text + image)

Transformer-based behavior modeling

Online learning / streaming updates

Quantized student encoder for edge deployment

9. License

This project is for research and educational purposes.
