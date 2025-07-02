
# Cocktail Consumer Insights Dashboard

An interactive Streamlit dashboard to explore a synthetic consumer survey on a new cocktail launch
in the Indian market. The app provides:

1. **Data Visualisation** – 10 descriptive insights with charts.
2. **Classification** – KNN, Decision Tree, Random Forest, Gradient Boosting with metrics, confusion matrix,
   ROC curves, and prediction on new uploads.
3. **Clustering** – K-Means with elbow method, interactive cluster selection, persona summaries,
   and data download.
4. **Association Rule Mining** – Apriori algorithm with user‑adjustable parameters.
5. **Regression** – Linear, Ridge, Lasso, Decision Tree regression models for spend prediction.

## Getting Started

```bash
# Clone your GitHub repo (or use Streamlit Cloud's direct Git integration)
git clone https://github.com/<your‑username>/<your‑repo>.git
cd <your‑repo>

# Install dependencies
pip install -r requirements.txt

# Run locally
streamlit run streamlit_app.py
```

## File Structure

```
.
├── streamlit_app.py   # Main Streamlit app
├── synthetic_cocktail_survey.csv  # Sample dataset (3,000 rows)
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

## Deployment

1. Push the repository to GitHub.
2. Go to **Streamlit Cloud → New app**.
3. Select the repo and set **`streamlit_app.py`** as the entry‑point file.
4. Deploy!

Enjoy exploring the cocktail consumer insights 🚀
