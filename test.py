from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
import gradio as gr
#import parseToDoc
import unstructuredDirectoryLoader

directory_loader = unstructuredDirectoryLoader.CustomDirectoryLoader(directory_path='C:\\Users\\kpriyank\\VSCodeRepo\\HFLocalRAG\\LocalRAG\\data\\samples')
docs = directory_loader.load()
#print(docs)
# Split into chunks
text_splitter = SemanticChunker(HuggingFaceEmbeddings())
documents = text_splitter.split_documents(docs)

# Instantiate the embedding model
embedder = HuggingFaceEmbeddings()

# Create the vector store and fill it with embeddings
vector = FAISS.from_documents(documents, embedder)
vector.save_local("faiss_index")
vector = FAISS.load_local("faiss_index", embedder, allow_dangerous_deserialization=True)

retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Define llm
llm = Ollama(model="mistral")

# Define the prompt
prompt = """
1. Use the following pieces of context to answer the question at the end.
2. If you don't know the answer, just say that "I don't know" but don't make up an answer on your own.\n
3. Keep the answer crisp and limited to 3,4 sentences.
4. Always end all responses with the document path that was used to generate the responses.

Context: {context}

Question: {question}

Helpful Answer:"""

QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt) 

llm_chain = LLMChain(
                  llm=llm, 
                  prompt=QA_CHAIN_PROMPT, 
                  callbacks=None, 
                  verbose=True)

document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
)

combine_documents_chain = StuffDocumentsChain(
                  llm_chain=llm_chain,
                  document_variable_name="context",
                  document_prompt=document_prompt,
                  callbacks=None)
              
qa = RetrievalQA(
                  combine_documents_chain=combine_documents_chain,
                  verbose=True,
                  retriever=retriever,
                  return_source_documents=True)

def respond(question,history):
    return qa(question)["result"]

    #files1 = gr.Textbox("data/samples/Air-Asia-Ticket-Receipt.pdf  | Source: https://view.officeapps.live.com/op/view.aspx?src=https%3A%2F%2Ftemplaterepublic.com%2Fwp-content%2Fuploads%2FAir-Asia-Ticket-Receipt.doc&wdOrigin=BROWSELINK", label="Data1")
    #files2 = gr.Textbox("data/samples/diabetes.csv  | Source: https://www.kaggle.com/datasets/mathchi/diabetes-data-set", label="Data2")
    #files3 = gr.Textbox("data/samples/Mental Health Dataset.msg", label="Data3")
    #files4 = gr.Textbox("data/samples/release.txt", label="Data4")
    #files5 = gr.Textbox("data/samples/US_Declaration.pdf | Source: https://storage.googleapis.com/kagglesdsdata/datasets/1172535/1963863/US_Declaration.pdf?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20240618%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240618T185919Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=00b3091b8fec9aa9f8cb03cccc1f01903a1bee3b27f6b6d07743191f877247a1cb1907f568cb038717ed33d071b7877b49165d02599adea3b117b3d7da036595fc0382a2d5c86939e997df1834ecbc8d2714040db50b55a4df906f33a62015bae20e5a350dfbb08ccdc9e667ea335aad17592c3d752b2224a5cbc85fa78a82afcaa21a3af0e6c60317a8935b12a8d10844bbcf52f03981315c015d553aa45652287ab1684a4c927df9d062d951b46067a84bdf02643362b18b4effb0e207aedd9c8148a2ba0fdbf2ad57a1ca59b672819079415d7d1618445f7ae50dc6920ba3b0ef8688e6ba64b7bf11d32a5431aec0eba5cf6c370d2d42e2bc82f62684aadb", label="Data5")


gr.ChatInterface(
    respond,
    chatbot=gr.Chatbot(height=500),
    textbox=gr.Textbox(placeholder="Ask me any questions. The Data is store in ./data/samples/ folder", container=False, scale=7),
    title="RAG Chatbot",
    cache_examples=True,
    retry_btn=None,

).launch(share = False)
