import os
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool, StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
from pydantic import BaseModel
from typing import List, Optional

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
model = init_chat_model("gpt-4o-mini", model_provider="openai", max_tokens=2000, temperature=0.3)
memory = MemorySaver()
matplotlib.use('Agg')

db = SQLDatabase.from_uri(os.getenv("DB_URI"))
query_sql_tool = QuerySQLDatabaseTool(db=db)

repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with print(...).",
    func=PythonREPL().run,
)

class PlotInput(BaseModel):
    x: List[int]
    y: List[float]
    graph_folder: str
    filename: str
    title: Optional[str] = "Line Plot"
    xlabel: Optional[str] = "X"
    ylabel: Optional[str] = "Y"

def generate_line_plot(x, y, graph_folder, filename, title="Line Plot", xlabel="X", ylabel="Y"):
    dir = os.path.dirname(graph_folder)
    if dir and not os.path.exists(dir):
        os.makedirs(dir)

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    path = os.path.join(graph_folder, filename)
    plt.savefig(path)
    plt.close()

def generate_line_plot_wrapper(inputs: PlotInput) -> str:
    generate_line_plot(inputs.x, inputs.y, inputs.graph_folder, inputs.filename, inputs.title, inputs.xlabel, inputs.ylabel)
    return f"Graph generated: {inputs.graph_folder}/{inputs.filename}"

graph_plot_tool = StructuredTool.from_function(
    func=generate_line_plot_wrapper,
    input_schema=PlotInput,
    description=(
        "Use this tool to generate line plots. Required keys: "
        "'x' (list of x values), 'y' (list of y values). Optional: "
        "'filename', 'title', 'xlabel', 'ylabel'."
    )
)

tools = [query_sql_tool, graph_plot_tool, repl_tool]

agent_executor = create_react_agent(model, tools, checkpointer=memory)

db_description = """
The finanal_db database has one main table: `company_data`, with the following structure:

### Company Metadata and Financial Data
1. company_id: Unique identifier for the company.
2. ticker: The stock ticker symbol of the company.
3. company_name: Full name of the company.
4. country: The country where the company is based.
5. industry_code: Numeric code representing the company's industry classification.
6. year: The fiscal year of the financial data.

### Base Financial Data
7. current_assets: The company's current assets.
8. total_assets: The company's total assets.
9. cash: The company's cash on hand.
10. current_debt: The company's current debt.
11. long_term_debt: The company's long-term debt.
12. invested_capital: The company's total invested capital.
13. total_liabilities: The company's total liabilities.
14. cost_of_goods_sold: The cost of goods sold.
15. ebit: Earnings before interest and taxes.
16. ebitda: Earnings before interest, taxes, depreciation, and amortization.
17. eps: Earnings per share.
18. net_income: The company's net income.
19. total_revenue: The company's total revenue.
20. income_taxes: Income taxes.
21. interest_expense: The company's interest expenses.
22. capital_expenditures: The company's capital expenditures.
23. net_cash_flow_financing: Net cash flow from financing activities.
24. net_cash_flow_investing: Net cash flow from investing activities.
25. net_cash_flow_operating: Net cash flow from operating activities.
26. common_shares_outstanding: Number of common shares outstanding.
27. total_equity: total shareholder's equity.
28. dividends_per_share: Dividends paid per share.
29. market_value: The company's market capitalization value.
30. price: The company's closing stock price.

### Derived Financial Metrics
31. revenue_growth: Year-over-year revenue growth percentage.
32. eps_growth: Year-over-year earnings per share growth.
33. dividend_growth: Year-over-year growth in dividends per share.
34. net_profit_margin: Net income as a percentage of total revenue.
35. operating_margin: Operating profit as a percentage of total revenue.
36. gross_margin: Gross profit as a percentage of total revenue.
37. return_on_assets(ROA): Net income as a percentage of total assets.
38. return_on_equity(ROE): Net income as a percentage of total equity.
39. return_on_invested_capital(ROIC): Net income as a percentage of (total assets - total liabilities).
40. free_cash_flow(FCF): Net cash flow from operations minus capital expenditures.
41. free_cash_flow_margin(FCF margin): Free cash flow as a percentage of total revenue.
42. debt_to_equity(D/E): (Current debt + Long-term debt) divided by total equity.
43. debt_to_assets(D/A) : (Current debt + Long-term debt) divided by total assets.
44. price_to_earnings_ratio(P/E) : Closing stock price divided by earnings per share.
45. price_to_book_ratio(P/B) : Market value divided by total equity.
46. price_to_share_ratio(P/S) : Market value divided by common shares outstanding.
47. EV_to_EBITDA_ratio(EV/EBITDA): (Market value + total liabilities - cash) divided by EBITDA.

Use the `ticker` column to identify companies when a company name or stock symbol is mentioned.
"""

system_message = SystemMessage(content="""
You are a financial analysis agent designed to interact with a SQL database containing company financial data.

If the question is related to the financial data of companies (for example, asking about revenue, earnings, or financial ratios), you will query the 'company_data' table in the database.
If the question is general in nature (such as asking for the capital of a country or historical events), you will provide an answer using your internal knowledge base, drawing from common knowledge and general sources.

{db_description}

Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.

If the query results include time series data or numerical values suitable for visualization 
(e.g., revenue over years), use the `graph_plot_tool` to generate a 
relevant chart (line chart, bar chart, etc.). 

If a graph is not appropriate, return only the text-based answer.
""".format(
    dialect="MySQL",
    top_k=5,
    db_description=db_description
))

def query_agent(user_input: str):
    user_message = HumanMessage(content=user_input)
    config = {"configurable": {"thread_id": "thread-001"}}
    full_response = ""

    for step in agent_executor.stream({"messages": [system_message, user_message]}, config, stream_mode="values"):
        if step["messages"]:
            step["messages"][-1].pretty_print()
        if step["messages"] and isinstance(step["messages"][-1], AIMessage):
            chunk = step["messages"][-1].content
            full_response += chunk
    return full_response
