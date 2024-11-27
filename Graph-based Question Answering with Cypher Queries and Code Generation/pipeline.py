# pipeline.py
from langchain.chains import GraphCypherQAChain
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
import openai
from config import OPENAI_API_KEY
from neo4j_utils import connect_to_neo4j

# Set up your OpenAI API key
openai.api_key = OPENAI_API_KEY

def run_pipeline(question):
    # Neo4j Database Connection
    url = "bolt://localhost:7687"
    username = ""
    password = ""
    graph = connect_to_neo4j(url, username, password)

    # Cypher Query Generation Template
    CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher query to retrieve the 'prompt' property of a node.
    Instructions:
    1. Use only the provided node label and property in the schema.
    2. Do not use any other node labels or properties that are not provided.
    3. Construct a Cypher query specifically designed to retrieve the 'prompt' property of a node.

    Schema:
    {schema}

    Example Cypher Query:
    # Retrieve the 'prompt' property of a PieChart node
    MATCH (pc:PieChart)
    RETURN pc.prompt

    # Retrieve the 'prompt' property of an EDA node
    MATCH (eda:EDA)
    RETURN eda.prompt

    Note: Do not include any explanations or apologies in your responses.
    Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
    Do not include any text except the generated Cypher statement.

    The question is:
    {question}"""

    CYPHER_GENERATION_PROMPT = PromptTemplate(
        input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE
    )

    # GraphCypherQAChain
    chain = GraphCypherQAChain.from_llm(
        llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo'),
        graph=graph,
        verbose=True,
        return_direct=True,
        cypher_prompt=CYPHER_GENERATION_PROMPT,
    )

    result = chain.run(question)

    # Extract the prompt sentence
    prompt_sentence = result[0]['bc.prompt']

    # OpenAI Interaction and Code Generation
    prompt = f"Code a Python script to {prompt_sentence}"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # GPT-3.5-turbo model
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
    )

    generated_code = response['choices'][0]['message']['content']

    # Print or use the generated code as needed
    print(generated_code)

# main script
from pipeline import run_pipeline

result = run_pipeline("how to do eda?")
