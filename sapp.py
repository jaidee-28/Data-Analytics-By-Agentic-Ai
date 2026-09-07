"""
Streamlit Sales Analyzer
------------------------
Upload a sales CSV/Excel file. The app:
  1. Auto-detects date, numeric (revenue/quantity/price), and
     categorical (product/region/customer) columns.
  2. Renders a set of relevant plots automatically.
  3. Sends the computed summary stats to GPT-4o and shows concrete,
     data-grounded suggestions for improving sales.

Run with:
    streamlit run streamlit_app.py
"""
import os

from langchain_openai import ChatOpenAI
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

st.set_page_config(page_title="Sales Analyzer", layout="wide")
st.title("📊 Sales Data Analyzer")
st.caption("Upload your sales data to get automatic plots and AI-generated suggestions.")

# ---- API key --------------------------------------------------------------

api_key = st.secrets["api_key"]
api_kye= os.getenv("api_key")

# ---- upload -----------------------------------------------------------------
uploaded_file = st.file_uploader("Upload sales data (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("Upload a file to get started. Expected columns might include things like "
            "date, product, category, region, quantity, price, revenue/sales — "
            "but the app will adapt to whatever columns you have.")
    st.stop()

# ---- load ---------------------------------------------------------------
def read_csv_robust(file):
    """Try a sequence of encodings and separators; CSVs from Excel/Windows
    are often cp1252/latin-1 rather than utf-8, and sometimes semicolon-
    or tab-separated rather than comma-separated."""
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin-1"]
    last_err = None
    for enc in encodings:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc, sep=None, engine="python")
        except Exception as e:
            last_err = e
            continue
    raise last_err


try:
    if uploaded_file.name.lower().endswith(".csv"):
        df = read_csv_robust(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(
        f"Could not read file: {e}\n\n"
        "Try opening it in Excel/Google Sheets and re-saving as "
        "'CSV UTF-8 (Comma delimited)', then re-upload."
    )
    st.stop()

st.success(f"Loaded {df.shape[0]:,} rows and {df.shape[1]} columns.")
with st.expander("Preview data", expanded=False):
    st.dataframe(df.head(20))
    st.write("**Column types:**")
    st.dataframe(df.dtypes.astype(str).rename("dtype"))

# ---- auto-detect column roles ---------------------------------------------
def find_col(candidates, columns):
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        for lower, orig in cols_lower.items():
            if cand in lower:
                return orig
    return None


date_col = find_col(["date", "order_date", "time", "month", "year"], df.columns)
revenue_col = find_col(["revenue", "sales", "amount", "total", "price"], df.columns)
quantity_col = find_col(["quantity", "qty", "units"], df.columns)
product_col = find_col(["product", "item", "sku"], df.columns)
category_col = find_col(["category", "segment", "type"], df.columns)
region_col = find_col(["region", "state", "country", "city", "location"], df.columns)
customer_col = find_col(["customer", "client"], df.columns)

# try to parse date column
if date_col:
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if df[date_col].isna().all():
            date_col = None
    except Exception:
        date_col = None

numeric_cols = df.select_dtypes(include="number").columns.tolist()
if revenue_col is None and numeric_cols:
    revenue_col = numeric_cols[0]

st.divider()
st.subheader("Detected fields")
st.write(
    f"- **Date:** `{date_col}`  \n"
    f"- **Revenue/Sales:** `{revenue_col}`  \n"
    f"- **Quantity:** `{quantity_col}`  \n"
    f"- **Product:** `{product_col}`  \n"
    f"- **Category:** `{category_col}`  \n"
    f"- **Region:** `{region_col}`  \n"
    f"- **Customer:** `{customer_col}`"
)
st.caption("Wrong guess? Rename your columns to something like 'date', 'revenue', "
           "'product', 'region', etc. and re-upload for better detection.")

# ---- plots ------------------------------------------------------------------
st.divider()
st.subheader("📈 Plots")

summary_notes = []  # text snippets fed to the LLM later

col1, col2 = st.columns(2)

# 1. Revenue over time
if date_col and revenue_col:
    with col1:
        ts = df.dropna(subset=[date_col]).set_index(date_col)[revenue_col].resample("ME").sum()
        fig, ax = plt.subplots()
        ts.plot(ax=ax, marker="o")
        ax.set_title(f"{revenue_col} over time (monthly)")
        ax.set_ylabel(revenue_col)
        st.pyplot(fig)
    if len(ts) >= 2:
        change = (ts.iloc[-1] - ts.iloc[0]) / max(abs(ts.iloc[0]), 1e-9) * 100
        summary_notes.append(
            f"Monthly {revenue_col} went from {ts.iloc[0]:,.0f} to {ts.iloc[-1]:,.0f} "
            f"({change:+.1f}% over the period)."
        )

# 2. Top products by revenue
if product_col and revenue_col:
    with col2:
        top_products = df.groupby(product_col)[revenue_col].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots()
        top_products.plot(kind="barh", ax=ax)
        ax.invert_yaxis()
        ax.set_title(f"Top 10 {product_col} by {revenue_col}")
        st.pyplot(fig)
    summary_notes.append(
        f"Top product by {revenue_col}: {top_products.index[0]} "
        f"({top_products.iloc[0]:,.0f}). "
        f"Bottom of top-10: {top_products.index[-1]} ({top_products.iloc[-1]:,.0f})."
    )

col3, col4 = st.columns(2)

# 3. Revenue by category
if category_col and revenue_col:
    with col3:
        cat_rev = df.groupby(category_col)[revenue_col].sum().sort_values(ascending=False)
        fig, ax = plt.subplots()
        ax.pie(cat_rev, labels=cat_rev.index, autopct="%1.1f%%")
        ax.set_title(f"{revenue_col} share by {category_col}")
        st.pyplot(fig)
    summary_notes.append(
        f"By {category_col}, {cat_rev.index[0]} leads with "
        f"{cat_rev.iloc[0] / cat_rev.sum() * 100:.1f}% of total {revenue_col}, "
        f"while {cat_rev.index[-1]} trails at {cat_rev.iloc[-1] / cat_rev.sum() * 100:.1f}%."
    )

# 4. Revenue by region
if region_col and revenue_col:
    with col4:
        region_rev = df.groupby(region_col)[revenue_col].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots()
        region_rev.plot(kind="bar", ax=ax)
        ax.set_title(f"{revenue_col} by {region_col}")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)
    summary_notes.append(
        f"Top {region_col}: {region_rev.index[0]} ({region_rev.iloc[0]:,.0f} {revenue_col})."
    )

col5, col6 = st.columns(2)

# 5. Top customers
if customer_col and revenue_col:
    with col5:
        top_cust = df.groupby(customer_col)[revenue_col].sum().sort_values(ascending=False).head(10)
        fig, ax = plt.subplots()
        top_cust.plot(kind="barh", ax=ax)
        ax.invert_yaxis()
        ax.set_title(f"Top 10 {customer_col} by {revenue_col}")
        st.pyplot(fig)
    repeat_share = (df[customer_col].value_counts().gt(1).sum() / df[customer_col].nunique() * 100)
    summary_notes.append(
        f"{repeat_share:.1f}% of customers appear more than once in the data. "
        f"Top customer: {top_cust.index[0]} ({top_cust.iloc[0]:,.0f} {revenue_col})."
    )

# 6. Correlation heatmap for numeric columns
if len(numeric_cols) >= 2:
    with col6:
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots()
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha="right")
        ax.set_yticklabels(numeric_cols)
        fig.colorbar(im)
        ax.set_title("Correlation between numeric columns")
        st.pyplot(fig)

# 7. Quantity distribution
if quantity_col:
    fig, ax = plt.subplots()
    df[quantity_col].plot(kind="hist", bins=20, ax=ax)
    ax.set_title(f"Distribution of {quantity_col}")
    st.pyplot(fig)

# ---- key numbers table ------------------------------------------------------
st.divider()
st.subheader("🔢 Key numbers")
metrics = {}
if revenue_col:
    metrics["Total " + revenue_col] = f"{df[revenue_col].sum():,.0f}"
    metrics["Average " + revenue_col] = f"{df[revenue_col].mean():,.2f}"
if quantity_col:
    metrics["Total " + quantity_col] = f"{df[quantity_col].sum():,.0f}"
if product_col:
    metrics["Unique " + product_col] = df[product_col].nunique()
if customer_col:
    metrics["Unique " + customer_col] = df[customer_col].nunique()
st.table(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))

# ---- AI suggestions ---------------------------------------------------------
st.divider()
st.subheader("💡 AI Suggestions to Improve Sales")

if st.button("Generate suggestions"):
    if not api_key:
        st.error("Add your OpenAI API key in the sidebar first.")
    else:
        with st.spinner("Analyzing your data..."):
            client = ChatOpenAI(
                              openai_api_key=api_key,
                              openai_api_base="https://openrouter.ai/api/v1",
                              model_name="nvidia/nemotron-3-ultra-550b-a55b:free",
                              temperature=0.2,
                              max_tokens=2000,
                              streaming=True
                            )
            context = "\n".join(f"- {note}" for note in summary_notes) or "No detailed patterns detected."
            prompt = f"""You are a sales analytics consultant. Based on the following
computed facts from a company's sales data, give 5-7 concrete, specific,
actionable suggestions to improve sales. Reference the actual numbers given.
Avoid generic advice ("improve marketing") — tie each suggestion to a pattern
in the data below.

Facts from the data:
{context}

Key numbers:
{pd.DataFrame(metrics.items(), columns=["Metric", "Value"]).to_string(index=False)}
"""
            try:
                response = client.invoke([{"role": "user", "content": prompt}])
                st.markdown(response.content)
            except Exception as e:
                st.error(f"Error calling OpenAI: {e}")
else:
    st.caption("Click the button above to get AI-generated recommendations based on the plots and numbers above.")
