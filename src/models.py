from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.chains import HypotheticalDocumentEmbedder
from langchain.vectorstores import Chroma
from src.constants import SECRET_KEY

class Extractor:

    def __init__(self, paths, segment_length = 512, top_k_documents = 5) -> None:
        self.docs = []
        self.segment_length = segment_length
        self.top_k = top_k_documents
        self.load_docs(paths)

    def load_docs(self, paths):
        for path in paths:
            loader = TextLoader(path)
            document = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=self.segment_length, chunk_overlap=0)
            self.docs += text_splitter.split_documents(document)



class tf_idfExtractor(Extractor):
    def __init__(self, paths, segment_length = 512, top_k_documents = 5) -> None:
        super().__init__(paths, segment_length, top_k_documents)
        self.name = 'tf_idf'

    def infer_relevant_docs(self, query):
        return self.get_top_k_articles(query)

    def get_top_k_articles(self, query):

        # Initialize a vectorizer that removes English stop words
        vectorizer = TfidfVectorizer(analyzer="word", stop_words='english')

        docasstrings = []
        for doc in self.docs:
            docasstrings.append(doc.page_content)

        # Create a corpus of query and documents and convert to TFIDF vectors
        query_and_docs = [query] + docasstrings
        matrix = vectorizer.fit_transform(query_and_docs)

        # Holds our cosine similarity scores
        scores = []

        # The first vector is our query text, so compute the similarity of our query against all document vectors
        for i in range(1, len(query_and_docs)):
            scores.append(cosine_similarity(matrix[0], matrix[i])[0][0])

        # Sort list of scores and return the top k highest scoring documents
        sorted_list = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        top_doc_indices = [x[0] for x in sorted_list[:self.top_k]]
        top_docs = [self.docs[x] for x in top_doc_indices]
        
        return top_docs
    
class dpr_Extractor(Extractor):
    def __init__(self, paths, segment_length = 512, top_k_documents = 5) -> None:
        super().__init__(paths, segment_length, top_k_documents)
        self.name = 'dpr'
    
    def infer_relevant_docs(self, query):
        embeddings = HuggingFaceEmbeddings()
        docsearch = OpenSearchVectorSearch.from_documents(self.docs, embeddings, opensearch_url="http://localhost:9200")
        return docsearch.similarity_search(query, k=self.top_k)
    
class hyde_Extractor(Extractor):
    def __init__(self, paths, segment_length = 512, top_k_documents = 5) -> None:
        super().__init__(paths, segment_length, top_k_documents)
        self.name = 'hyde'
    
    def infer_relevant_docs(self, query):
        base_embeddings = OpenAIEmbeddings(openai_api_key=SECRET_KEY)
        llm = OpenAI(openai_api_key=SECRET_KEY)
        embeddings = HypotheticalDocumentEmbedder.from_llm(llm, base_embeddings, "web_search")
        result = embeddings.embed_query(query)
        docsearch = OpenSearchVectorSearch.from_documents(self.docs, embeddings, opensearch_url="http://localhost:9200", bulk_size=1500)
        return docsearch.similarity_search(query, k=self.top_k)