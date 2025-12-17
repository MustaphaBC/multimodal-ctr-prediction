# Multimodal CTR Prediction (MM-CTR)

This project implements a **two-stage multimodal Click-Through Rate (CTR) prediction system** combining offline multimodal embedding learning and online CTR prediction with user behavior modeling.

The architecture follows the diagram below:

- **Task 1 (Offline)**: Multimodal Embedding via knowledge distillation from a frozen **CLIP ViT** teacher to a lightweight student encoder.
- **Task 2 (Online)**: CTR prediction using a **DIN-style attention mechanism** over user behavior history.

---

## 1. Project Overview

### Objective
Predict the probability that a user will click on a target item by jointly modeling:

- Visual representation of the target item  
- Sequential user behavior history  
- Attention-based interest matching (DIN)

This design is aligned with the **MM-CTR Challenge (WWW 2025 – EReL)** setting, emphasizing **efficiency, modularity, and multimodal learning**.

---

## 2. System Architecture

### Task 1: Multimodal Embedding (Offline)

**Goal:** Learn compact, high-quality visual embeddings for items.

**Pipeline:**
1. Input target item image  
2. Frozen **CLIP ViT** extracts high-dimensional visual features (teacher)  
3. **Student Encoder** learns via knowledge distillation  
4. Output: **128-dimensional embedding** per item  

**Key Characteristics:**
- CLIP parameters are frozen  
- Student encoder is lightweight and deployment-friendly  
- Embeddings are precomputed and stored for online use  

---

### Task 2: CTR Prediction (Online)

**Goal:** Predict CTR in real time using learned embeddings.

**Pipeline:**
1. User behavior sequence (past clicked/viewed items)  
2. Target item embedding used as attention query  
3. **DIN Attention Layer** computes interest relevance  
4. Weighted sum of historical embeddings  
5. **MLP** predicts CTR score  

**Advantages:**
- Captures dynamic and personalized user interests  
- Efficient inference through offline–online separation  

---

## 3. Installation

### 3.1 Environment Setup

- **Python version:** `>= 3.9`

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate   # Linux / MacOS
venv\Scripts\activate    # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. Training Workflow

### Step 1: Offline Embedding Training

```bash
python training/train_embedding.py
```

**Outputs:**
- Trained student encoder weights  
- Precomputed 128-dimensional item embeddings  

---

### Step 2: CTR Model Training

```bash
python training/train_ctr.py
```

**Inputs:**
- User behavior sequences  
- Precomputed item embeddings  

**Outputs:**
- Trained **DIN + MLP** CTR prediction model  

---

## 5. Inference

```bash
python inference/predict_ctr.py \
  --user_id 123 \
  --item_id 456
```

**Returns:**
- Predicted CTR probability score  

---

## 6. Evaluation Metrics

- **AUC** (Area Under the ROC Curve)  
- **LogLoss**  
- **CTR Calibration Error**  

---

## 7. Key Design Choices

- **Knowledge Distillation:** Reduces inference cost while preserving CLIP semantic power  
- **DIN Attention:** Enables fine-grained personalized interest modeling  
- **Offline–Online Separation:** Improves scalability, latency, and deployment efficiency  

---

## 8. Future Improvements

- Multimodal fusion (image + text)  
- Transformer-based behavior sequence modeling  
- Online learning / streaming updates  
- Quantized student encoder for edge or mobile deployment  

---

## 9. License

This project is intended for **research and educational purposes only**.
