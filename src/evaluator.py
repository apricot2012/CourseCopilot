from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

def lines_to_relevant_strings(lines, paths, segment_length):
    docs = []
    for path in paths:
        loader = TextLoader(path)
        document = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=segment_length, chunk_overlap=0)
        docs += text_splitter.split_documents(document)
    contained = set()
    for doc in docs:
        for line in lines:
            if line in doc.page_content:
                contained.add(doc.page_content)

    return list(contained)

def percentage_contained(groundtruth, result):
    num = 0
    denum = len(groundtruth)
    for segment in groundtruth:
        if segment in result:
            num += 1
    return num / denum
