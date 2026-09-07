# Data-Analytics-By-Agentic-Ai
# 📊 Streamlit Sales Data Analyzer

An automated sales analytics dashboard built with **Streamlit**, **Pandas**, and **LangChain**. The application automatically detects dataset attributes, generates visual insights, compiles statistical summaries, and uses an LLM via OpenRouter to deliver data-grounded strategic sales recommendations.

---

## 🌟 Key Features

* **Smart Data Ingestion:** Robust multi-encoding engine handles `.csv` and `.xlsx`/`.xls` files generated across different platforms (Windows, Excel, Unix).
* **Automatic Column Detection:** Infers roles for dates, revenue, quantity, product, category, region, and customer identifiers dynamically.
* **Automated Visualizations:**
  * Monthly Revenue Trends (Line Chart)
  * Top 10 Products & Customers (Horizontal Bar Charts)
  * Category Revenue Share (Pie Chart)
  * Regional Performance Distribution (Vertical Bar Chart)
  * Correlation Heatmap (Numeric Variables)
  * Quantity Distribution (Histogram)
* **Key Metrics Summary:** Auto-aggregates totals, averages, and unique counts into a high-level KPI overview.
* **AI-Powered Recommendations:** Sends computed factual notes and metrics to an LLM to generate actionable, data-backed sales strategies.

---

## 🚀 Getting Started

### Prerequisites

Ensure you have Python 3.9+ installed on your system.

### 1. Installation

Clone the repository and install the dependencies:
