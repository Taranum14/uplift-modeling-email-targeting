# Who Should Get the Email? Uplift Modeling for Marketing Targeting

**Live dashboard:**

An end-to-end uplift analysis of a randomized email campaign (64,000 customers). Instead of predicting who is likely to buy, it estimates **for whom the email actually changes the outcome**, then turns that into a profit decision: who to email, and when emailing everyone is the better choice.

<!-- Add a screenshot or GIF of the dashboard: save it as assets/dashboard.png and uncomment the next line -->
<!-- ![Dashboard](assets/dashboard.png) -->

## The business problem

Marketing teams usually target customers who are most likely to buy. Many of those customers would have bought anyway, so the email budget is partly wasted. Uplift modeling separates customers into four groups:

| Group | Buys if emailed | Buys if not emailed | Worth emailing? |
|---|---|---|---|
| Persuadables | Yes | No | Yes |
| Sure things | Yes | Yes | No |
| Lost causes | No | No | No |
| Sleeping dogs | No | Yes | Never |

Nobody is ever observed both ways, so the effect has to be estimated from a randomized experiment.

## Data

[Hillstrom MineThatData E-Mail Analytics challenge](https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html): 64,000 customers randomly assigned to a men's email, a women's email, or no email. Features: recency, past spend, spend segment, product history, region type, new vs existing customer, and purchase channel. Outcomes: website visit, conversion, and spend. The two email arms are pooled into one treatment group.

## Key results

| Finding | Result |
|---|---|
| Randomization check | Treated and control groups balanced on every feature checked |
| Visit rate | 10.6% without email, 16.7% with email (+6.1 points, 95% CI 5.5 to 6.6) |
| Conversion rate | 0.57% without email, 1.07% with email |
| Spend per customer | $0.65 without email, $1.25 with email (about +$0.60) |
| Variation across segments | Fairly uniform: visit lift of about 5 to 8 points across channel, tenure, region and past spend |

**Model comparison** (Qini AUC over 8 random stratified train/test splits, higher is better):

| Model | Mean | Std |
|---|---|---|
| Response model (baseline) | 0.0176 | 0.0099 |
| **S-learner** | **0.0296** | **0.0050** |
| T-learner | 0.0158 | 0.0091 |
| Class transformation | 0.0138 | 0.0062 |

The S-learner was the most stable model. The T-learner scored second best on a single split, but that lead disappeared over eight splits. The S-learner's gain over the baseline is modest, and its range still overlaps the baseline's, so I do not claim a conclusive win.

## The decision: when does targeting matter?

Profit on the 19,200-customer held-out test set, assuming a 30% gross margin on spend:

| Cost per email | Email everyone | Best uplift policy | Share emailed | Random targeting (average) |
|---|---|---|---|---|
| $0.05 | $2,205 | $2,205 | 100% | $2,205 |
| $0.10 | $1,245 | $1,245 | 100% | $1,245 |
| $0.15 | $285 | $560 | 15% | $324 |
| $0.20 | -$675 | $416 | 15% | $58 |
| $0.30 | -$2,595 | $128 | 15% | -$38 |
| $0.40 | -$4,515 | $26 | 5% | -$134 |

**Takeaway:** the average email adds about $0.55 of spend, which is about $0.165 of margin. Below that cost per email, emailing everyone is optimal and targeting adds nothing. Above it, a small high-uplift slice stays profitable while a blanket send loses money. Profit estimates carry roughly +/- $100 of noise at this sample size.

The dashboard recomputes this live as you change margin, cost per email and list size, so its figures differ slightly from the table above (finer grid, scaled list size).

## Approach

1. **Randomization check.** Compared treated and control groups on all features, then estimated the overall effect with a confidence interval.
2. **Segment exploration.** Measured email lift within each customer segment to see whether the effect varies at all.
3. **Baselines and uplift models.** A response model plus three uplift learners (S-learner, T-learner, class transformation), all using LightGBM.
4. **Honest evaluation.** Qini curves, Qini AUC and uplift@k, repeated over 8 random splits, because a single split crowned the wrong model.
5. **Money.** Estimated incremental spend per customer, clipped extreme predictions, and compared targeting policies against observed treated-versus-control spend in held-out data.
6. **Dashboard.** Streamlit app with a campaign planner, decile diagnostics, a cost-and-margin heatmap and a customer scorer.

## Limitations

- Margin and cost per email are assumptions, which is why they are sliders.
- The best share of the list is chosen on the same test data used to measure profit, so the peak is slightly optimistic.
- Spend has a heavy tail: a few large purchases drive much of the apparent variation between customers.
- One campaign, with outcomes observed over two weeks. Long-term effects are not captured.
- Pooling the men's and women's emails hides any difference between them.

## Project structure

```
uplift-app/
├── app.py                      Streamlit dashboard
├── requirements.txt
├── .streamlit/config.toml      Light theme
├── data/test_scored.parquet    Held-out customers with model scores
├── models/                     S-learner (LightGBM text format) and column list
└── notebooks/
    └── 01_uplift_analysis.ipynb  Full analysis
```

## Run locally

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To rerun the analysis, open the notebook and install `scikit-uplift`, `scikit-learn` and `jupyter` as well.

## Tech stack

Python, pandas, NumPy, scikit-learn, scikit-uplift, LightGBM, Plotly, Streamlit.

## Next steps

- Treat the men's and women's emails as separate treatments.
- Choose the targeted share on a validation split and report profit on untouched data.
- Test whether the Criteo uplift dataset (about 14 million rows) shows stronger heterogeneity.

## Author

Taranum Patel | [LinkedIn](www.linkedin.com/in/taranum-patel) | taranum14patel@gmail.com
