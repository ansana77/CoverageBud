import os
import logging
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser


load_dotenv()
key = os.environ["GROQ_API_KEY"]

# chatpdf-system-prompt: """"You are an intelligent assistant designed to analyze and #compare two or more insurance brochures provided in PDF format. Your task is to: 
# Thoroughly analyze both brochures, extracting key details about coverage, benefits, #limitations, and pricing.
# Evaluate the effectiveness of each insurance plan based on the user's specific needs #and preferences.
# Ensure that all of the user’s questions are fully addressed with accurate, clear, and relevant information from the brochures.
# If there is missing or unclear information in the brochures, clarify this and offer insights on what the user should consider when making a decision.
# Provide well-structured, concise, and easy-to-understand responses that directly help the user compare the plans effectively.""""

chatpdf_system_prompt = """"You are an intelligent assistant designed to analyze and compare insurance brochures provided in PDF format. Your task is to:

1. Thoroughly analyze both brochures, extracting key details about coverage, benefits, limitations, and pricing.
2. Evaluate the effectiveness of each insurance plan based on the user's specific needs and preferences.
3. Ensure that all of the user’s questions are fully addressed with accurate, clear, and relevant information from the brochures.
4. If there is missing or unclear information in the brochures, clarify this and offer insights on what the user should consider when making a decision.
5. Provide well-structured, concise, and easy-to-understand responses that directly help the user compare the plans effectively.
Respond in this format:
<First Document Name>
<First Document Relevant Information>
<Second Document Name>
<Second Document Relevant Information>
<Final Verdict and Response>
"""

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

#function to perform search in the vectorstore to retrieve 3 similar neighbors
def retrieve_chunks(vectorstore, uuid_dict, user_query):
    grouped_by_uuid = {}
    for uuid in uuid_dict:
        print(f"Retrieving for: {uuid}\n\n")
        neighbor = vectorstore.similarity_search_with_score(user_query, filter={"pdf_uuid": uuid}, k=5)
        grouped_by_uuid[uuid] = [document[0].page_content for document in neighbor]
    return grouped_by_uuid

    
def reply_generator(base_query, uuid_dict, retrieved_docs):
    messages = [
    ("system", chatpdf_system_prompt),
    ]
    for idx, doc_uuid in enumerate(uuid_dict):
        document_prompt = f"Here is the information for Document {idx + 1}. The name of the document is {uuid_dict[doc_uuid]}"
        document_content = "\n".join(retrieved_docs[doc_uuid])
        document_prompt += document_content
        messages.append(
            ("human", document_prompt)
        )
    messages.append(
        ("human", base_query)
    )
    result = llm.invoke(messages)
    return result

if __name__ == "__main__":
    pass