import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import streamlit as st


# os.getenv("OPENAI_API_KEY")
os.getenv("TAVILY_API_KEY")

OPENAI_API_KEY = st.sidebar.text_input("OpenAI API Key", type="password")

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


search_instructions = """You will be given sets of questions and answers with explanation for the exam. 
{sets}

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the exam.
        
First, analyze the sets of the questions and answers with solutions.

Convert this sets into a well-structured web search query to check if the answers and solutions are correct and updated."""

# Web search tool
from langchain_community.tools.tavily_search import TavilySearchResults

tavily_search = TavilySearchResults(max_results=5)


def search_web(state: State):
    """Retrieve docs from web search"""

    preps = state["preps"]

    system_message = search_instructions.format(sets=preps)
    structured_llm = llm.with_structured_output(SearchQuery)
    search_query = structured_llm.invoke([SystemMessage(content=system_message)])

    # Search docs
    search_docs = tavily_search.invoke(search_query.search_query)

    formatted_search_docs = ["\n--------------\n".join(doc) for doc in search_docs]

    return {"context": formatted_search_docs}


def search_rag(state: State):
    """Retrieve docs from vector store"""

    from langchain_openai import OpenAIEmbeddings
    from langchain_core.vectorstores import InMemoryVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = InMemoryVectorStore(embeddings)

    pdf_file = state["pdf_file"]
    preps = state["preps"]
    # load pdf files
    loader = PyPDFLoader(pdf_file)
    docs = loader.load()
    # docs = []
    # for page in loader.lazy_load():
    #     docs.append(page.page_content)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    splitted_docs = text_splitter.split_documents(docs)

    vector_store.add_documents(splitted_docs)

    # Retrieve docs
    retrieved_docs = vector_store.similarity_search(str(preps))

    return {"context": [retrieved_docs]}


explanation_instructions = """You are an expert in the field of given subject.

Here is area of focus: {subject}.

Here are the sets of question & answer with solution:
Sets: {sets}

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
        subject=subject, sets=preps, context=context
    )
    structured_llm = llm.with_structured_output(Perspectives)

    refined_preps = structured_llm.invoke(
        [SystemMessage(content=system_message)]
        + [HumanMessage(content="Reproduce Q&As with more accurate explanation.")]
    )

    # Append it to state
    return {"preps": refined_preps.preps}


def should_continue(state: State):
    """Return the next node to execute"""

    # Check if human feedback
    human_feedback = state.get("human_feedback", None)
    if human_feedback:
        return "create_prep"

    # Otherwise
    return "refine_explanation"


# Add nodes and edges
builder = StateGraph(State)
builder.add_node("create_prep", create_prep)
builder.add_node("human_feedback_check", human_feedback_check)
builder.add_node("search_rag", search_rag)
builder.add_node("search_web", search_web)
builder.add_node("refine_explanation", refine_explanation)

builder.add_edge(START, "create_prep")
builder.add_edge("create_prep", "search_rag")
builder.add_edge("create_prep", "search_web")
builder.add_edge("search_rag", "refine_explanation")
builder.add_edge("search_web", "refine_explanation")
builder.add_edge("refine_explanation", "human_feedback_check")
builder.add_conditional_edges(
    "human_feedback_check", should_continue, ["create_prep", "refine_explanation"]
)
builder.add_edge("refine_explanation", END)

memory = MemorySaver()
graph = builder.compile(interrupt_before=["human_feedback_check"], checkpointer=memory)


# View
# display(Image(graph.get_graph(xray=1).draw_mermaid_png())) -> Jupyter lab

# ------------------------------------------------------------------ #
### FRONTEND SECTION && RUN ###
import os
import tempfile

thread = {"configurable": {"thread_id": "1"}}

# current_path = os.getcwd()
# path = current_path + pdf_file

st.title("GENERATE Q&As w/ solution")


with st.form("form"):
    subject = st.text_input(
        "Enter subject:",
        "",
    )

    num = st.number_input("Introduce the number of Q&A sets.", min_value=1)

    feedback = st.text_area("Put your preferences(difficulty, focus etc)", "")

    pdf_file = st.file_uploader("Choose your pdf file", type="pdf")
    submitted = st.form_submit_button("GENERATE Q&A")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        try:
            temp_file.write(pdf_file.getbuffer())
            pdf_file = temp_file.name
        except:
            print("")
    # st.success(f"Temporary file saved at: {temp_file_path}")

if not OPENAI_API_KEY.startswith("sk-"):
    st.warning("Please enter your OpenAI API key!", icon="⚠")
if submitted and OPENAI_API_KEY.startswith("sk-"):
    # Run the graph until the first interruption
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
    with st.spinner("Generating..."):
        for event in graph.stream(
            {"subject": subject, "pdf_file": pdf_file, "num": num},
            thread,
            stream_mode="values",
        ):
            # Review
            preps = event.get("preps")
            if preps:
                for prep in preps:
                    print(f"Question: {prep.question}")
                    print(f"Answer: {prep.answer}")
                    print(f"Explanation: {prep.explanation}")
                    # print("-" * 50)
                    # st.write("Question: ")
                    # st.info(prep.question)

        graph.update_state(
            thread,
            {"human_feedback": feedback},
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

        for event in graph.stream(None, thread, stream_mode="updates"):
            print(event)

        final_state = graph.get_state(thread)
        qas = final_state.values.get("preps")

        for qa in qas:
            qas_show.append(qa)
            # print("Final Q&As")
            # print(f"Question: {qa.question}")
            # print(f"Answer: {qa.answer}")
            # print(f"Explanation: {qa.explanation}")
            # print("-" * 50)
            st.write("Question: ")
            st.info(prep.question)
            st.write("Answer")
            st.info(prep.answer)
            st.write("Explanation")
            st.info(prep.explanation)
