# Event-Driven State Space Model (SSM) with PAD-Style Training

## 1. Problem Setup

We observe an event sequence:

\[
\mathcal{D} = \{(t_i, e_i)\}_{i=1}^{T}
\]

where:

- \( t_i \in \mathbb{R}^+ \): timestamp  
- \( e_i \in \mathcal{E} \): discrete event  

Define inter-event time:

\[
\Delta t_i = t_i - t_{i-1}
\]

---

## 2. Event Representation

Each event is embedded into a continuous vector space:

\[
u_i = \phi(e_i) \in \mathbb{R}^d
\]

Optionally include time encoding:

\[
u_i = \phi(e_i) \oplus \psi(\Delta t_i)
\]

where:

- \( \phi \): event embedding  
- \( \psi \): time encoding function  

---

## 3. Continuous-Time State Space Model

Let \( z(t) \in \mathbb{R}^h \) denote the latent state.

### 3.1 Continuous evolution between events

\[
\frac{dz(t)}{dt} = A z(t)
\]

Solution:

\[
z(t_i^-) = \exp(A \Delta t_i)\, z(t_{i-1})
\]

---

### 3.2 Discrete update at event time

\[
z_i = z(t_i^-) + B u_i
\]

---

### 3.3 Combined update rule

\[
z_i = \exp(A \Delta t_i)\, z_{i-1} + B u_i
\]

where:

- \( A \in \mathbb{R}^{h \times h} \): transition matrix  
- \( B \in \mathbb{R}^{h \times d} \): input matrix  

---

## 4. Dual-Branch Architecture (PAD-style)

We define two latent states:

### 4.1 Anomaly branch

\[
h_i = \exp(A_f \Delta t_i)\, h_{i-1} + B_f u_i
\]

### 4.2 Precursor (PoA) branch

\[
z_i = \exp(A_g \Delta t_i)\, z_{i-1} + B_g u_i
\]

---

### 4.3 Parameter sharing

Optionally:

\[
A_f = A_g = A_c
\]

This enables shared temporal dynamics.

---

## 5. Output Heads

### 5.1 Anomaly prediction

\[
y_i^{(a)} = \sigma(W_a h_i + b_a)
\]

---

### 5.2 Precursor prediction

\[
y_i^{(p)} = \sigma(W_p z_i + b_p)
\]

where:

- \( \sigma \): sigmoid function  

---

## 6. Window-Based Formulation

Partition sequence into windows:

\[
w_k = \{(t_i, e_i)\}_{i \in \mathcal{I}_k}
\]

Define:

- \( y_k \): anomaly label for window \( w_k \)

Model produces:

- \( y_k^{(a)} \): anomaly prediction  
- \( y_k^{(p)} \): precursor prediction  

---

## 7. Loss Functions

### 7.1 Anomaly loss

\[
\mathcal{L}_a = \mathrm{CE}(y_k^{(a)}, y_k)
\]

---

### 7.2 Knowledge distillation (PoA loss)

\[
\mathcal{L}_{kd} = \mathrm{CE}(y_k^{(p)}, y_{k+1}^{(a)})
\]

---

### 7.3 Total loss

\[
\mathcal{L} = \mathcal{L}_a + \lambda \mathcal{L}_{kd}
\]

---

## 8. Training Algorithm

### Input

- Training dataset \( \mathcal{D} \)  
- Maximum iterations \( K \)

---

### Initialize

- \( A_f, A_g, B_f, B_g, W_a, W_p \)

---

### Loop

For \( k = 1 \) to \( K \):

1. Sample window pair \( (w_k, w_{k+1}) \)

2. Compute embeddings:
   \[
   u_i = \phi(e_i)
   \]

3. Forward pass (window \( w_k \)):

   For each event \( i \in w_k \):

   \[
   \begin{aligned}
   h_i &= \exp(A_f \Delta t_i)\, h_{i-1} + B_f u_i \\
   z_i &= \exp(A_g \Delta t_i)\, z_{i-1} + B_g u_i
   \end{aligned}
   \]

4. Compute outputs:

   \[
   y_k^{(a)} = \sigma(W_a h_T)
   \]
   \[
   y_k^{(p)} = \sigma(W_p z_T)
   \]

---

1. Forward pass (next window \( w_{k+1} \)):

   \[
   y_{k+1}^{(a)} = \sigma(W_a h'_T)
   \]

---

1. Compute losses:

   \[
   \mathcal{L}_a = \mathrm{CE}(y_k^{(a)}, y_k)
   \]
   \[
   \mathcal{L}_{kd} = \mathrm{CE}(y_k^{(p)}, y_{k+1}^{(a)})
   \]

---

1. Total loss:

   \[
   \mathcal{L} = \mathcal{L}_a + \lambda \mathcal{L}_{kd}
   \]

---

1. Update parameters using gradient descent.

---

## 9. Summary

This model combines:

- Continuous-time evolution:
  \[
  \exp(A \Delta t)
  \]

- Event-driven updates:
  \[
  B u_i
  \]

- Dual-task learning:
  - anomaly detection  
  - precursor prediction  

- Knowledge distillation:
  - future anomaly supervision  

---

## 10. Key Properties

- Handles irregular event timing  
- No interpolation required  
- Supports long-term memory via \( A \)  
- Captures event-driven dynamics via \( B \)  
- Enables early anomaly detection (PoA)  
