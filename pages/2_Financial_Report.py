import streamlit as st
import yfinance as yf
from datetime import datetime
import pandas as pd
from typing import TypedDict, Dict, List
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import uuid
import os

# Initialize Groq LLM
llm = ChatGroq(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    api_key=st.secrets["GROQ_API_KEY"]
)

# Define categories
categories = {
    "Basic Info": [
        "symbol", "shortName", "longName", "sector", "industry", 
        "country", "exchange", "currency", "quoteType"
    ],
    "Price Info": [
        "currentPrice", "open", "dayHigh", "dayLow", "previousClose", 
        "fiftyTwoWeekHigh", "fiftyTwoWeekLow", "volume", "averageVolume"
    ],
    "Valuation": [
        "marketCap", "enterpriseValue", "trailingPE", "forwardPE", 
        "priceToSalesTrailing12Months", "priceToBook", "enterpriseToRevenue", 
        "enterpriseToEbitda"
    ],
    "Financial Metrics": [
        "totalRevenue", "grossMargins", "ebitdaMargins", "operatingMargins", 
        "profitMargins", "totalCash", "totalDebt", "debtToEquity", 
        "returnOnAssets", "returnOnEquity", "freeCashflow"
    ],
    "Dividends": [
        "dividendRate", "dividendYield", "payoutRatio", 
        "lastDividendDate", "exDividendDate"
    ],
    "Analysts": [
        "targetMeanPrice", "recommendationMean", "recommendationKey", 
        "numberOfAnalystOpinions", "earningsGrowth", "revenueGrowth"
    ]
}

# Define state for LangGraph
class FinancialState(TypedDict):
    ticker: str
    info: Dict
    news: List[Dict]
    history: pd.DataFrame
    report: str
    valuation_data: Dict
    margins_data: Dict
    market_data: Dict

# Nodes for LangGraph workflow
def fetch_data(state: FinancialState) -> FinancialState:
    stock = yf.Ticker(state["ticker"])
    state["info"] = stock.info
    state["news"] = stock.news
    state["history"] = stock.history(period="5y")
    return state

def process_financial_data(state: FinancialState) -> FinancialState:
    info = state["info"]
    ticker = state["ticker"]
    valuation_data = {}
    margins_data = {}
    market_data = {}

    result = f"# 📊 Financial Report: `{info.get('shortName', ticker)}` ({ticker})\n\n"

    # Process each category and display as a markdown table
    for category, fields in categories.items():
        table_data = []
        for field in fields:
            if field in info and info[field] is not None:
                value = info[field]
                if 'Date' in field and isinstance(value, int):
                    value = datetime.fromtimestamp(value).strftime('%Y-%m-%d')
                elif 'Margins' in field or 'Yield' in field or 'Ratio' in field or 'Growth' in field:
                    if isinstance(value, (int, float)):
                        value = f"{value:.2%}"
                elif field in ['marketCap', 'enterpriseValue', 'totalRevenue', 'totalCash', 
                              'totalDebt', 'freeCashflow'] and isinstance(value, (int, float)):
                    if value >= 1_000_000_000:
                        value = f"${value/1_000_000_000:.2f} B"
                    elif value >= 1_000_000:
                        value = f"${value/1_000_000:.2f} M"
                    else:
                        value = f"${value:,.2f}"

                table_data.append([field.replace(field[0], field[0].upper()), f"`{value}`"])

                # Categorize data for charts
                if category == "Valuation" and isinstance(value, (int, float)):
                    if field in ["trailingPE", "forwardPE", "priceToSalesTrailing12Months", 
                               "priceToBook", "debtToEquity"]:
                        valuation_data[field] = value
                if category == "Financial Metrics" and isinstance(value, (int, float)):
                    if field in ["grossMargins", "ebitdaMargins", "operatingMargins", 
                               "profitMargins"]:
                        margins_data[field] = value
                    if field in ["marketCap", "enterpriseValue", "totalRevenue", 
                               "totalCash", "totalDebt"]:
                        market_data[field] = value

        if table_data:
            result += f"## 📈 {category}\n\n"
            result += "| Metric | Value |\n|--|-------|\n"
            result += "\n".join([f"| {row[0]} | {row[1]} |" for row in table_data]) + "\n\n"

    state["report"] = result
    state["valuation_data"] = valuation_data
    state["margins_data"] = margins_data
    state["market_data"] = market_data
    return state

def generate_analysis(state: FinancialState) -> FinancialState:
    prompt = PromptTemplate(
        input_variables=["report", "ticker"],
        template="Based on the following financial data for {ticker}, provide:\n1. A financial analysis report in 5 structured paragraphs with some data in tables.\n2. A conclusion with investment risk value (range 1-9) in `` code-styled subtitle format, and outlook (1-2 sentences)\n\n{report}"
    )

    chain = prompt | llm
    response = chain.invoke({
        "report": state["report"],
        "ticker": state["ticker"]
    }).content

    state["report"] += f"## 🔍 `Analysis & Conclusion`\n\n{response}\n"
    return state

# Build LangGraph workflow
workflow = StateGraph(FinancialState)
workflow.add_node("fetch_data", fetch_data)
workflow.add_node("process_financial_data", process_financial_data)
workflow.add_node("generate_analysis", generate_analysis)

workflow.add_edge("fetch_data", "process_financial_data")
workflow.add_edge("process_financial_data", "generate_analysis")
workflow.add_edge("generate_analysis", END)

workflow.set_entry_point("fetch_data")
graph = workflow.compile()

# Streamlit app
st.set_page_config(page_title="Financial Agent", layout="wide")
st.subheader("📈 Financial Agent: :grey-background[Stock] Analysis 🏦", divider='grey')
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, NVDA):", "PLTR").upper()

if st.button("✨ Generate Analysis Report", use_container_width=True):
    with st.spinner("✨ Collection Data and generationg report..."):
        # Execute workflow
        initial_state = FinancialState(
            ticker=ticker,
            info={},
            news=[],
            history=pd.DataFrame(),
            report="",
            valuation_data={},
            margins_data={},
            market_data={}
        )

        result = graph.invoke(initial_state)

        # Display company logo and key metrics
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image(f"https://logos.stockanalysis.com/{ticker.lower()}.svg", width=60)
        with col2:
            cols = st.columns(4)
            metrics = [
                ("`Price`", result["info"].get("currentPrice", 0), "$"),
                ("`Market Cap`", result["info"].get("marketCap", 0)/1_000_000_000, "B$"),
                ("`P/E Ratio`", result["info"].get("trailingPE", 0), ""),
                ("`P/S Ratio`", result["info"].get("priceToSalesTrailing12Months", 0), "")
            ]
            for i, (label, value, unit) in enumerate(metrics):
                with cols[i]:
                    st.metric(
                        label=label,
                        value=f"{value:.2f}{unit}" if isinstance(value, (int, float)) else value
                    )

        # Display company summary
        with st.expander("🏢 Company Overview"):
            st.write(result["info"].get("longBusinessSummary", "No summary available"))

        # Display report with HTML rendering
        #st.markdown(result["report"], unsafe_allow_html=True)
        st.markdown(result["report"])

        # Display charts and tables in wide containers
        with st.container():
            if result["valuation_data"]:
                st.subheader("📊 Valuation Ratios")
                # Convert dictionary to markdown table
                valuation_table = "| Metric | Value |\n|------|-------|\n"
                for key, value in result["valuation_data"].items():
                    valuation_table += f"| {key.replace(key[0], key[0].upper())} | {value:.2f} |\n"
                st.markdown(valuation_table)
                st.bar_chart(pd.DataFrame.from_dict(result["valuation_data"], orient="index", columns=["Value"]), use_container_width=True)

            if result["margins_data"]:
                st.subheader("📊 Margin Metrics")
                # Convert dictionary to markdown table
                margins_table = "| Metric | Value |\n|------|-------|\n"
                for key, value in result["margins_data"].items():
                    margins_table += f"| {key.replace(key[0], key[0].upper())} | {value:.2%} |\n"
                st.markdown(margins_table)
                st.bar_chart(pd.DataFrame.from_dict(result["margins_data"], orient="index", columns=["Value"]), use_container_width=True)

            if result["market_data"]:
                st.subheader("📊 Market Metrics")
                # Convert dictionary to markdown table
                market_table = "| Metric | Value |\n|------|-------|\n"
                for key, value in result["market_data"].items():
                    if value >= 1_000_000_000:
                        formatted_value = f"${value/1_000_000_000:.2f} B"
                    elif value >= 1_000_000:
                        formatted_value = f"${value/1_000_000:.2f} M"
                    else:
                        formatted_value = f"${value:,.2f}"
                    market_table += f"| {key.replace(key[0], key[0].upper())} | {formatted_value} |\n"
                st.markdown(market_table)
                st.bar_chart(pd.DataFrame.from_dict(result["market_data"], orient="index", columns=["Value"]), use_container_width=True)

        if not result["history"].empty:
            st.subheader("📈 5-Year Price History")
            st.line_chart(result["history"]["Close"], use_container_width=True)

        # Display news with proper error handling
        st.subheader("📰 Latest News")
        if result["news"] and isinstance(result["news"], list):
            for item in result["news"]:
                # Ensure required fields exist, provide defaults if missing
                title = item.get('title', 'No Title Available')
                link = item.get('link', '#')
                publisher = item.get('publisher', 'Unknown Publisher')
                related_tickers = item.get('relatedTickers', [])
                publish_time = item.get('providerPublishTime', None)

                # Convert timestamp to readable format
                try:
                    publish_date = pd.to_datetime(publish_time, unit='s').strftime('%Y-%m-%d %H:%M:%S') if publish_time else 'Unknown Date'
                except (ValueError, TypeError):
                    publish_date = 'Unknown Date'

                # Render news item
                st.markdown(f"> 📄 **{title}** . 🔗 [Link]({link})")
                st.text(f"📰 {publisher} | 📆 {publish_date} | 🌐 {', '.join(related_tickers) or 'N/A'}")
        else:
            st.info("No news available for this ticker at the moment.")

