from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from src.constants import SECRET_KEY

class Enquirer:
    def __init__(self) -> None:
        self.prompt_template ="""Use the following pieces of context to answer the question at the end. 
            Refer back to the context and explain how you arrived at such answer.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            {context}

            Question: {question}
            Answer:"""
        
    def perform_qa(self, context, query):
        PROMPT = PromptTemplate(template=self.prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(OpenAI(temperature=0, openai_api_key=SECRET_KEY), chain_type="stuff", prompt=PROMPT)
        return chain({"input_documents": context, "question": query}, return_only_outputs=True)