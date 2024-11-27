# neo4j_utils.py
from langchain.graphs import Neo4jGraph

def connect_to_neo4j(url, username, password):
    return Neo4jGraph(url=url, username=username, password=password)