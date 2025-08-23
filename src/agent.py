import os
import uuid
from fastapi import FastAPI
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool, StructuredTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib
from pydantic import BaseModel
from typing import List, Optional
matplotlib.use('Agg')

try:
    db_uri = f'postgresql+psycopg://{os.getenv("DB_USER")}:{os.getenv("DB_PASSWORD")}@{os.getenv("DB_HOST")}:5432/{os.getenv("DB_NAME")}'
    db = SQLDatabase.from_uri(db_uri)
    query_sql_tool = QuerySQLDatabaseTool(db=db)
except Exception as e:
    print(f'Could not connect to PostgreSQL DB. Check DB_URI or make sure server is running. Error: {e}')
    exit(1)

repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with print(...).",
    func=PythonREPL().run,
)

class PlotInput(BaseModel):
    x: List[int|str]
    y: List[float]
    graph_folder: str
    filename: str
    title: Optional[str] = "Plot"
    xlabel: Optional[str] = "X"
    ylabel: Optional[str] = "Y"

class MultiPlotInput(BaseModel):
    x: List[int|str]
    y: List[List[float]]
    labels: List[str]
    graph_folder: str
    filename: str
    title: Optional[str] = "Plot"
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

def generate_multiline_plot(x, y, graph_folder, filename, title="Multiline Plot", xlabel="X", ylabel="Y", labels=None):
    dir = os.path.dirname(graph_folder)
    if dir and not os.path.exists(dir):
        os.makedirs(dir)

    plt.figure()
    for i, y_values in enumerate(y):
        plt.plot(x, y_values, marker="o", label=labels[i])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()

    path = os.path.join(graph_folder, filename)
    plt.savefig(path)
    plt.close()

def generate_bar_plot(x, y, graph_folder, filename, title="Bar Plot", xlabel="X", ylabel="Y"):
    dir = os.path.dirname(graph_folder)
    if dir and not os.path.exists(dir):
        os.makedirs(dir)

    plt.figure()
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45,ha='right')
    plt.grid(axis='y')

    plt.tight_layout()

    path = os.path.join(graph_folder, filename)
    plt.savefig(path)
    plt.close()

def generate_pie_chart(x, y, graph_folder, filename, title="Pie Chart"):
    dir = os.path.dirname(graph_folder)
    if dir and not os.path.exists(dir):
        os.makedirs(dir)

    plt.figure(figsize=(8, 5))
    
    plt.pie(y, labels=x, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.tight_layout()

    path = os.path.join(graph_folder, filename)
    plt.savefig(path)
    plt.close()

def generate_line_plot_wrapper(inputs: PlotInput) -> str:
    generate_line_plot(inputs.x, inputs.y, inputs.graph_folder, inputs.filename, inputs.title, inputs.xlabel, inputs.ylabel)
    return f"Graph generated: {inputs.graph_folder}/{inputs.filename}"

def generate_multiline_plot_wrapper(inputs: MultiPlotInput) -> str:
    generate_multiline_plot(inputs.x, inputs.y, inputs.graph_folder, inputs.filename, inputs.title, inputs.xlabel, inputs.ylabel, inputs.labels)
    return f"Graph generated: {inputs.graph_folder}/{inputs.filename}"

def generate_bar_plot_wrapper(inputs: PlotInput) -> str:
    generate_bar_plot(inputs.x, inputs.y, inputs.graph_folder, inputs.filename, inputs.title, inputs.xlabel, inputs.ylabel)
    return f"Graph generated: {inputs.graph_folder}/{inputs.filename}"

def generate_pie_chart_wrapper(inputs: PlotInput) -> str:
    generate_pie_chart(inputs.x, inputs.y, inputs.graph_folder, inputs.filename, inputs.title)
    return f"Graph generated: {inputs.graph_folder}/{inputs.filename}"

graph_line_plot_tool = StructuredTool.from_function(
    func=generate_line_plot_wrapper,
    input_schema=PlotInput,
    description=(
        "Use this tool to generate line plots. "
        "Example use case: the question is about a trend over time for one company."
        "Required keys: 'x' (list of x values), 'y' (list of y values), 'graph_folder' (folder to save graph in)."
        "Optional: 'filename', 'title (title of the figure)', 'xlabel (name of horizontal axis)', 'ylabel (name of vertical axis)'."
    )
)

graph_multiline_plot_tool = StructuredTool.from_function(
    func=generate_multiline_plot_wrapper,
    input_schema=MultiPlotInput,
    description=(
        "Use this tool to generate line plots for M multiple datasets. "
        "Example use case: the question is about comparing multiple time series."
        "Required keys: 'x' (list of x-values), 'y' (nested list of y-values), 's' (list of M labels), 'graph_folder' (folder to save graph in)."
        "Optional: 'filename', 'title (title of the figure)', 'xlabel (name of horizontal axis)', 'ylabel (name of vertical axis)'."
    )
)

graph_bar_plot_tool = StructuredTool.from_function(
    func=generate_bar_plot_wrapper,
    input_schema=PlotInput,
    description=(
        "Use this tool to generate bar plots."
        "Example use case: the question is about different companies in a specific year."
        "Required keys: 'x' (list of x values), 'y' (list of y values), 'graph_folder' (folder to save graph in)."
        "Optional: 'filename', 'title (title of the figure)', 'xlabel (name of horizontal axis)', 'ylabel (name of vertical axis)'."
    )
)

graph_pie_chart_tool = StructuredTool.from_function(
    func=generate_pie_chart_wrapper,
    input_schema=PlotInput,
    description=(
        "Use this tool to generate pie charts."
        "Example use case: the question is about the breakdown of a quantity into different aspects."
        "Required keys: 'x' (list of labels), 'y' (list of values), 'graph_folder' (folder to save graph in)."
        "Optional: 'filename', 'title (title of the figure)'."
    )
)

llm = init_chat_model(os.getenv("OPENAI_MODEL_NAME"), 
                      model_provider="openai", 
                      temperature=0.)

graph = create_react_agent(llm, 
                           tools=[repl_tool, query_sql_tool, graph_line_plot_tool, graph_multiline_plot_tool, graph_bar_plot_tool, graph_pie_chart_tool], 
                           checkpointer=InMemorySaver())

def create_system_message():
    with open('prompt.txt','r') as f:
        latent_system_message = f.read()

    with open('db_description.txt','r') as f:
        db_description = f.read()
    system_message = SystemMessage(content=latent_system_message.format(
        dialect=db.dialect,
        top_k=5,
        db_description=db_description
    ))
    print(system_message)
    return system_message

system_message = None
thread_id = uuid.uuid4().hex[:8]
config = {"configurable": {"thread_id": thread_id}}

async def send_init_prompt(app:FastAPI):
    global graph
    global config
    global system_message
    system_message = create_system_message()
    response = await graph.ainvoke({"messages" :[system_message]}, config)
    
    # Store the init response for injection into HTML
    app.state.init_response = response["messages"][-1].content
    
    # Open the gate for queries
    app.state.init_prompt_done.set()

def query_agent(user_input: str):
    global graph
    global config
    graph_png_filename = f"graph/graph_{uuid.uuid4().hex[:8]}.png"
    # put commands that cannot be baked into the prompt here
    preamble = SystemMessage(f"""PREAMBLE:
                             If a graph is generated, save it as {graph_png_filename}. Do not plt.show(). 
                             Display the image with <img src={graph_png_filename} max-width=100% height=auto>.
                             """)

    user_message = HumanMessage(content=user_input)

    for step in graph.stream({"messages": [preamble, user_message]}, config, stream_mode="values"):
        if step["messages"]:
            step["messages"][-1].pretty_print()
        if step["messages"] and isinstance(step["messages"][-1], AIMessage):
            chunk = step["messages"][-1].content
            yield chunk