import streamlit as st
import yfinance as yf
from typing import Annotated, List, Dict, Any, Tuple, Optional, Type
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langgraph.prebuilt import ToolNode, tools_condition

# Set page configuration
st.set_page_config(
    page_title="Financial Assistant",
    page_icon="🏦",
    layout="wide",
)


# Display header
st.subheader("🏦 Financial Assistant Chatbot", divider='orange')


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.graph = None


def yfinance_info_tool(ticker: str) -> str:
    """Get detailed financial informations about a stock ticker using yfinance."""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        # If info is empty or None, return an error message
        if not info:
            return f"Could not retrieve information for ticker: {ticker}"

        # Organize financial data by categories
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

        result = f"## Financial Information for {info.get('shortName', ticker)}\n\n"

        for category, fields in categories.items():
            category_data = []
            for field in fields:
                if field in info and info[field] is not None:
                    value = info[field]
                    # Format dates
                    if 'Date' in field and isinstance(value, int):
                        from datetime import datetime
                        value = datetime.fromtimestamp(value).strftime('%Y-%m-%d')
                    # Format percentages
                    elif 'Margins' in field or 'Yield' in field or 'Ratio' in field or 'Growth' in field:
                        if isinstance(value, (int, float)):
                            value = f"{value:.2%}"
                    # Format currency values
                    elif field in ['marketCap', 'enterpriseValue', 'totalRevenue', 'totalCash', 
                                  'totalDebt', 'freeCashflow'] and isinstance(value, (int, float)):
                        if value >= 1_000_000_000:
                            value = f"${value/1_000_000_000:.2f}B"
                        elif value >= 1_000_000:
                            value = f"${value/1_000_000:.2f}M"
                        else:
                            value = f"${value:,.2f}"
                    category_data.append(f"- **{field}**: {value}")

            if category_data:
                result += f"### {category}\n" + "\n".join(category_data) + "\n\n"

        return result

    except Exception as e:
        return f"Error retrieving data for {ticker}: {str(e)}"


# Define the function to create our LangGraph
def create_financial_assistant_graph():
    # Define the state
    class State(TypedDict):
        messages: Annotated[list, add_messages]

    # Initialize the graph builder
    graph_builder = StateGraph(State)

    # Create the yfinance tool
    yfinance_tool = Tool(
        name="stock_info",
        description="Get detailed financial information about a stock by providing its ticker symbol",
        func=yfinance_info_tool,
    )

    # Initialize the LLM with Groq
    llm = ChatGroq(
        api_key=st.secrets["GROQ_API_KEY"],
        model_name="meta-llama/llama-4-scout-17b-16e-instruct", #"llama3-70b-8192",
        temperature=0.1,
    )

    # Add system message to help the LLM understand its role
    system_message = """You are a helpful financial assistant with access to stock information through the yfinance API.
    When users ask about stocks, use the stock_info tool by providing the ticker symbol.
    Always provide clear explanations and insights based on the financial data.
    If you're not sure about a ticker symbol, ask for clarification.
    Present financial information in a well-organized and easy-to-understand format.
    Format in markdown, emojies for better readability.
    """

    # Add tools to the LLM
    tools = [yfinance_tool]
    llm_with_tools = llm.bind_tools(tools)

    # Define the chatbot node
    def chatbot(state: State):
        messages = state["messages"]
        # Add system message to the start if it doesn't exist
        if not any(isinstance(msg, SystemMessage) for msg in messages):
            messages = [SystemMessage(content=system_message)] + messages

        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    # Create tool node
    tool_node = ToolNode(tools=tools)

    # Add nodes to the graph
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", tool_node)

    # Add conditional edges
    graph_builder.add_conditional_edges(
        "chatbot",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        }
    )

    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")

    # Start with the chatbot node
    graph_builder.add_edge(START, "chatbot")

    # Compile the graph
    return graph_builder.compile()


# Initialize the graph when app first loads
if st.session_state.graph is None:
    st.session_state.graph = create_financial_assistant_graph()

# Display the chat history
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# Get user input
user_input = st.chat_input("Ask about a stock or financial information...")

# Process user input
if user_input:
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=user_input))

    # Display the user message
    with st.chat_message("user"):
        st.write(user_input)

    # Get response from graph
    with st.spinner("Thinking..."):
        response = st.session_state.graph.invoke({"messages": st.session_state.messages})
        st.session_state.messages = response["messages"]

    # Display the assistant's response
    with st.chat_message("assistant"):
        st.markdown(st.session_state.messages[-1].content)


# Add a sidebar with instructions
with st.sidebar:
    st.header("How to Use")
    st.markdown("""
    ### Example Questions:
    - What's the current price of AAPL?
    - Tell me about Tesla stock (TSLA)
    - What are the financial metrics for Microsoft (MSFT)?
    - Compare the PE ratios of NVDA and AMD
    - What's the dividend yield for JNJ?

    ### Tips:
    - Include the ticker symbol in your question
    - Ask for specific metrics or categories of information
    - The assistant can provide detailed financial data from Yahoo Finance
    """)

    st.divider()
    st.caption("Built with Streamlit, LangChain, Groq, and Yahoo Finance")

# Add some CSS to improve the appearance
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }

    .css-1y4p8pa {
        max-width: 1000px;
    }
</style>
""", unsafe_allow_html=True)
