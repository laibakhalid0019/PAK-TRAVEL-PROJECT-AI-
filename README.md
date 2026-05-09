# 🚌 PakTravel AI System
### AL2002 Artificial Intelligence | FAST NUCES Chiniot-Faisalabad | Spring 2026

---

## 📦 Files

| File | Description |
|------|-------------|
| `paktravel_complete.ipynb` | Full Jupyter Notebook — Parts 1–5 |
| `app.py` | Streamlit GUI application |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## 🚀 How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the Jupyter Notebook
```bash
jupyter notebook paktravel_complete.ipynb
```
Run all cells top to bottom (Kernel → Restart & Run All).

### Step 3 — Launch the GUI
```bash
streamlit run app.py
```
Opens in your browser at `http://localhost:8501`

---

## 🗂️ Project Structure

### Part 1 — Search Algorithms
- Road network (15 cities, 19 roads) as weighted adjacency dictionary
- UCS, A* (with Islamabad heuristic), Bidirectional Search
- Relay route planner (2-leg journeys via transfer cities)
- NetworkX visualisation with highlighted optimal path

### Part 2 — Logic-Based AI Legal Advisor
- AIMA PropKB with all 15 NHA traffic laws
- Ahmed's facts → pl_resolution inference (avoids 2²⁹ truth-table rows)
- Modus Ponens, Hypothetical Syllogism, Modus Tollens, Resolution Refutation

### Part 3 — CSP Bus Scheduler
- 8 routes, 10 buses, hard and soft constraints
- Plain Backtracking vs MRV vs MRV+AC3
- AC3 prunes Bus9 from R3 (R8 singleton domain)
- Ali's relay bus (R3) guaranteed by HC2

### Part 4 — ANN Delay Prediction
- 500 synthetic records; rule-based labels
- sklearn MLPClassifier (10→8 ReLU, Sigmoid) — ~98% accuracy
- NumPy ANN from scratch: forward pass, BCE loss, backprop, GD
- 5 test journeys including Ali's Sukkur→Multan bus

### Part 5 — K-Means Clustering
- 400 traveller records; 6 features
- K-Means k=3: Business / Family / Commuter
- Elbow method confirms k=3 optimal
- Scatter, radar, histogram visualisations

---

## 🎛️ GUI Features

| Tab | Features |
|-----|---------|
| 🏠 Home | Hero banner, story stages, stats, interactive Pakistan map |
| 🗺️ Route Finder | City selector, algorithm chooser, live map, relay planner |
| ⚖️ Legal Advisor | Checkboxes for driver facts, inference chain, fine bar chart |
| 📅 Bus Scheduler | CSP solver, AC3 domain table, Gantt chart, Ali's bus confirmation |
| 🧠 Delay Predictor | Sliders, gauge chart, confusion matrix, training loss curve |
| 👥 Traveller Types | Scatter, radar, elbow, histograms, profile table |

---

## 👥 Group Members
*(Add your names here)*

---
*FAST NUCES | AL2002 Artificial Intelligence | Spring 2026*
