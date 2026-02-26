import os
from pymongo import MongoClient
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class IngestionScript:
    def __init__(self):
        self.mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        self.database_name = os.getenv("DATABASE_NAME", "ltm_database")
        self.collection_name = os.getenv("RAG_COLLECTION_NAME", "academic_resources")
        self.embedding_provider = os.getenv("EMBEDDING_PROVIDER", "openai")
        self.embedding_model_name = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.hf_model_name = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

        self.client = MongoClient(self.mongodb_uri)
        self.db = self.client[self.database_name]
        self.collection = self.db[self.collection_name]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        self.embeddings_model = self._initialize_embeddings_model()

    def _initialize_embeddings_model(self):
        if self.embedding_provider == "openai":
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY is not set for OpenAI embeddings.")
            return OpenAIEmbeddings(openai_api_key=self.openai_api_key, model=self.embedding_model_name)
        elif self.embedding_provider == "huggingface":
            return HuggingFaceEmbeddings(model_name=self.hf_model_name)
        else:
            raise ValueError(f"Unsupported embedding provider: {self.embedding_provider}")

    def load_resources(self, file_path: str) -> List[str]:
        """Loads text content from a given file path."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        return [content] # For simplicity, treat the whole file as one document first

    def create_chunks(self, documents: List[str]) -> List[str]:
        """Splits documents into smaller chunks."""
        chunks = []
        for doc in documents:
            chunks.extend(self.text_splitter.split_text(doc))
        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generates embeddings for a list of text chunks."""
        return self.embeddings_model.embed_documents(texts)

    def ingest_data(self, file_path: str):
        """Orchestrates the ingestion process for academic resources."""
        print(f"Loading resources from {file_path}...")
        documents = self.load_resources(file_path)
        
        print(f"Creating chunks from {len(documents)} documents...")
        chunks = self.create_chunks(documents)
        print(f"Generated {len(chunks)} chunks.")

        print("Generating embeddings for chunks...")
        embeddings = self.generate_embeddings(chunks)
        print("Embeddings generated.")

        print("Storing chunks and embeddings in MongoDB...")
        data_to_insert = []
        for i, chunk in enumerate(chunks):
            data_to_insert.append({
                "chunk_id": i,
                "content": chunk,
                "embedding": embeddings[i],
                "source": file_path,
                "timestamp": datetime.now()
            })
        
        if data_to_insert:
            self.collection.insert_many(data_to_insert)
            print(f"Successfully ingested {len(data_to_insert)} chunks into MongoDB collection '{self.collection_name}'.")
        else:
            print("No data to ingest.")

if __name__ == "__main__":
    from datetime import datetime
    # Example usage:
    # Create a dummy resource file for demonstration
    dummy_content = """
    ## Academic Stress Management Techniques

    Academic stress is a common issue among students. It can manifest as anxiety, fatigue, and difficulty concentrating.
    Effective strategies include:
    1.  **Time Management:** Plan your study schedule, break tasks into smaller parts, and avoid procrastination.
    2.  **Mindfulness and Meditation:** Practice deep breathing exercises or guided meditation to calm your mind.
    3.  **Physical Activity:** Regular exercise can significantly reduce stress levels. Aim for at least 30 minutes most days of the week.
    4.  **Healthy Diet:** Nutritional food supports brain function and overall well-being.
    5.  **Adequate Sleep:** Prioritize 7-9 hours of sleep per night to improve focus and mood.
    6.  **Social Support:** Talk to friends, family, or mentors about your feelings.
    7.  **Seek Professional Help:** Don't hesitate to reach out to school counselors or therapists if stress becomes overwhelming.

    ## Peer Pressure and How to Handle It

    Peer pressure can influence academic choices and personal behavior. It's important to develop resilience.
    Strategies include:
    1.  **Self-Awareness:** Understand your values and what you stand for.
    2.  **Assertiveness:** Learn to say "no" confidently without feeling guilty.
    3.  **Choose Your Friends Wisely:** Surround yourself with positive influences.
    4.  **Build Self-Esteem:** Focus on your strengths and achievements.
    5.  **Seek Adult Guidance:** Talk to a trusted adult about peer pressure situations.
    
    ## Study Tips for Exams
    
    Preparing for exams effectively can reduce stress.
    1.  **Start Early:** Begin reviewing material well in advance of the exam date.
    2.  **Active Recall:** Test yourself frequently instead of just re-reading notes.
    3.  **Spaced Repetition:** Review information at increasing intervals over time.
    4.  **Practice Problems:** Work through past papers or practice questions.
    5.  **Form Study Groups:** Collaborate with peers to understand concepts better.
    """
    
    dummy_file_path = "academic_resources.txt"
    with open(dummy_file_path, "w", encoding="utf-8") as f:
        f.write(dummy_content)

    ingestor = IngestionScript()
    ingestor.ingest_data(dummy_file_path)

    # Clean up dummy file
    os.remove(dummy_file_path)

    # To verify ingestion, you can connect to MongoDB and check the collection:
    # from pymongo import MongoClient
    # client = MongoClient(os.getenv("MONGODB_URI", "mongodb://localhost:27017/"))
    # db = client[os.getenv("DATABASE_NAME", "ltm_database")]
    # collection = db[os.getenv("RAG_COLLECTION_NAME", "academic_resources")]
    # print(f"Documents in collection: {collection.count_documents({})}")
    # for doc in collection.find().limit(2):
    #     print(doc)
