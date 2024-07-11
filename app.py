import folium
from streamlit_folium import folium_static as st_folium
from llama_index.core import SimpleDirectoryReader
from llama_index.core.schema import TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores import VectorStoreQuery
import streamlit as st
from qdrant_client import QdrantClient
import os
from dotenv import load_dotenv

# Initialize Streamlit
st.set_page_config(layout="centered")

# Load environment variables
load_dotenv()

# QDRANT
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# Hugging Face
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")

# Load dataset
input_file = "us_campsites.csv"
documents = SimpleDirectoryReader(input_files=[input_file]).load_data()

# Create TextNode instances
nodes = []
for doc in documents:
    if hasattr(doc, 'latitude') and hasattr(doc, 'longitude'):
        latitude = doc.latitude
        longitude = doc.longitude
        
        node = TextNode(text=doc.text)  # Assuming 'text' is the attribute name in your dataset
        node.metadata = {"latitude": latitude, "longitude": longitude}
        nodes.append(node)

# Embed text chunks
for node in nodes:
    node_embedding = embed_model.get_text_embedding(
        node.get_content(metadata_mode="all")
    )
    node.embedding = node_embedding

# Vector Store with QDRANT
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
vector_store = QdrantVectorStore(client=client, collection_name="Cluster0")
vector_store.add(nodes)

# Define Retriever class
class Retriever:
    def __init__(self, similarity_top_k, query_mode):
        self.similarity_top_k = similarity_top_k
        self.query_mode = query_mode
        self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en")
        self.vector_store = vector_store

    def _retrieve(self, query_str):
        query_embedding = self.embed_model.get_text_embedding(query_str)

        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self.similarity_top_k,
            mode=self.query_mode,
        )

        query_result = self.vector_store.query(vector_store_query)

        class NodeWithScore:
            def __init__(self, node, score):
                self.node = node
                self.score = score

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score = query_result.similarities[index] if query_result.similarities else None
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores

# Initialize Retriever
similarity_top_k = 2
query_mode = "default"
retriever_instance = Retriever(similarity_top_k, query_mode)

# Streamlit app
def main():
    st.title("Geospatial Semantic Search and Interactive Map Display Tool")
    
    st.write("Enter the name of a place:")
    place_name = st.text_input("", "")

    if st.button("Search"):
        if place_name:
            matched_city = search_city(place_name)
            if matched_city:
                latitude = matched_city.get("latitude")
                longitude = matched_city.get("longitude")
                if latitude is not None and longitude is not None:
                    st.write(f"Retrieved location for {place_name}: Latitude - {latitude}, Longitude - {longitude}")
                    show_map(latitude, longitude, place_name)
                else:
                    st.write("Latitude or Longitude not found in metadata.")
            else:
                st.write("Place not found")

def search_city(place_name):
    response = retriever_instance._retrieve(place_name)
    if response:
        for node_with_score in response:
            node = node_with_score.node
            if "latitude" in node.metadata and "longitude" in node.metadata:
                latitude = node.metadata["latitude"]
                longitude = node.metadata["longitude"]
                return {"latitude": latitude, "longitude": longitude}
    return None

def show_map(latitude, longitude, place_name):
    if latitude is not None and longitude is not None:
        m = folium.Map(location=[latitude, longitude], zoom_start=16)
        folium.Marker([latitude, longitude], popup=place_name, tooltip=place_name).add_to(m)
        st_folium(m, width=700)

if __name__ == "__main__":
    main()
