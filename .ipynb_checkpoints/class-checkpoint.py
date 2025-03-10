import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

os.getenv("OPENAI_API_KEY")
os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

from typing import List, Annotated
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
import operator


class Prep(BaseModel):
    question: str = Field(description="Questions for the exam")
    answer: str = Field(description="Answer of the generated questions")
    explanation: str = Field(description="Explanation about the answer")

    @property
    def qas(self) -> str:
        return f"Question: {self.question}\nAnswer: {self.answer}\nExplanation:{self.explanation}\n"


# Graph state
class State(TypedDict):
    subject: str
    pdf_file: str
    preps: List[Prep]
    context: Annotated[list, operator.add]
    human_feedback: str
    num: int


class Perspectives(BaseModel):
    preps: List[Prep] = Field(
        description="Comprehensive list of Q&As with explanations."
    )


class SearchQuery(BaseModel):
    search_query: str = Field(description="Search query for retrieval.")


from langchain_community.document_loaders import PyPDFLoader
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from IPython.display import Image, display

instructions = """You are an expert cram school teacher/professor. 

Your goal is providing questions, answers and its explanations so students can prepare for the specific subject's exam. Follow these instructions carefully:

1. First, review the subject:
{subject}
        
2. Review the provided sets of documents.
{documents}
    
3. Determine probable contents to be imposed in the exam based upon the documents.
                    
4. Produce {num} sets of questions, answers and explanations."""


# Nodes
def create_prep(state: State):
    """Generate sets of questions & explanations"""

    pdf_file = state.get("pdf_file")
    subject = state["subject"]
    num = state["num"]

    # load pdf files
    loader = PyPDFLoader(pdf_file)
    docs = []
    for page in loader.lazy_load():
        docs.append(page.page_content)

    # Force structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = instructions.format(subject=subject, documents=docs, num=num)

    # Generate preps
    preps = structured_llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content="Generate sets of questions and its explanations.")]
    )
    # Write the list of preps to state
    return {"preps": preps.preps}


def human_feedback_check(state: State):
    """No-op node that should be interrupted on"""
    pass


search_instructions = SystemMessage(
    content=f"""You will be given questions and solutions for the exam. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the exam.
        
First, analyze the full content of the questions and solutions.

Convert this set of question and solution into a well-structured web search query"""
)

# Web search tool
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_search = TavilySearchResults(
    max_results=5, search_depth="advanced", include_images=True
)


def search_web(state: State):
    """Retrieve docs from web search"""

    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([search_instructions] + state["preps"])

    # Search docs
    search_docs = tavily_search.invoke(search_query.search_query)

    formatted_search_docs = [
        "\n--------------\n".join(doc.content) for doc in search_docs
    ]

    print("Tavily search result: ")
    print(formatted_search_docs)

    return {"context": [formatted_search_docs]}


def search_rag(state: State):
    """Retrieve docs from vector store"""

    from langchain_openai import OpenAIEmbeddings
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embeddings)

    docs = state["pdf_file"]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
    splitted_docs = text_splitter.split_documents(docs)

    vector_store.add_documents(splitted_docs)

    # Retrieve docs
    retrieved_docs = vector_store.similarity_search(state["preps"])

    return {"context": [retrieved_docs]}


explanation_instructions = """You are an expert in the field of given subject.

Here is area of focus: {subject}.

Here is the set of question & answer:
Question: {question}

Answer: {answer} 

You don't have to update the questions and answers if the information are accurate and adequate for user's demand.
        
You goal is to refine the explanation about the corresponding answer so everyone can undestand and the information gets updated.

To explain answers, use this context:
        
{context}

When explaning answers, follow these guidelines:
        
1. Use only the information provided in the context. 
        
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. You can include graphs or images in your explanation next to any relevant statements.

4. The explanation must be well structured which make sense visually."""


def refine_explanation(state: State):
    """Node to refine Q&As and explanation"""

    # Get state
    preps = state["preps"]
    subject = state["subject"]
    context = state["context"]

    # Answer question
    system_message = explanation_instructions.format(
        subject=subject, question=preps.question, answer=preps.solution, context=context
    )
    refined_preps = llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content="Reproduce Q&As with more accurate explanation.")]
    )

    # Append it to state
    return {"preps": refined_preps}


def should_continue(state: State):
    """Return the next node to execute"""

    # Check if human feedback
    human_feedback = state.get("human_feedback", None)
    if human_feedback:
        return "create_prep"

    # Otherwise
    return "search_web"


# Add nodes and edges
builder = StateGraph(State)
builder.add_node("create_prep", create_prep)
builder.add_node("human_feedback_check", human_feedback_check)
builder.add_node("search_rag", search_rag)
builder.add_node("search_web", search_web)
builder.add_node("refine_explanation", refine_explanation)

builder.add_edge(START, "create_prep")
builder.add_edge("create_prep", "human_feedback_check")
builder.add_conditional_edges(
    "human_feedback_check", should_continue, ["create_prep", "search_web", "search_rag"]
)
builder.add_edge("search_rag", "refine_explanation")
builder.add_edge("search_web", "refine_explanation")
builder.add_edge("refine_explanation", END)

memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_feedback_check"], checkpointer=memory)


# View
# display(Image(graph.get_graph(xray=1).draw_mermaid_png()))

# !Here streamlit part would come

subject = "css"
pdf_file = "class/css.pdf"

thread = {"configurable": {"thread_id": "1"}}

# Run the graph until the first interruption
for event in graph.stream(
    {"subject": subject, "pdf_file": pdf_file, "num": 10}, thread, stream_mode="values"
):
    # Review
    preps = event.get("preps", [])
    if preps:
        for prep in preps:
            print(f"Question: {prep.question}")
            print(f"Answer: {prep.answer}")
            print(f"Explanation: {prep.explanation}")
            print("-" * 50)

print("*" * 70)

graph.update_state(
    thread,
    {"human_feedback": "Add in more difficult questions and give me 15 questions"},
    as_node="human_feedback_check",
)

for event in graph.stream(None, thread, stream_mode="values"):
    # Review
    preps = event.get("preps", [])
    if preps:
        for prep in preps:
            print(f"Question: {prep.question}")
            print(f"Answer: {prep.answer}")
            print(f"Explanation: {prep.explanation}")
            print("-" * 50)

print("*" * 70)

for event in graph.stream(None, thread, stream_mode="updates"):
    print(event)

final_state = graph.get_state(thread)
qas = final_state.values.get("preps")

for qa in qas:
    print("Final Q&As")
    print(f"Question: {qa.question}")
    print(f"Answer: {qa.answer}")
    print(f"Explanation: {qa.explanation}")
    print("-" * 50)
