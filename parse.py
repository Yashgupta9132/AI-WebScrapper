from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import httpx

model = OllamaLLM(model="llama3")


template = (
    "You are tasked with extracting specific information from the following text content: {dom_content}. "
    "Please follow these instructions carefully: \n\n"
    "1. **Extract Information:** Only extract the information that directly matches the provided description: {parse_description}. "
    "2. **No Extra Content:** Do not include any additional text, comments, or explanations in your response. "
    "3. **Empty Response:** If no information matches the description, return an empty string ('')."
    "4. **Direct Data Only:** Your output should contain only the data that is explicitly requested, with no other text."
)


def parse_with_ollama(dom_chunks , parse_description):
    promt = ChatPromptTemplate.from_template(template)
    chain = promt | model 

    parsed_results = []

    for i, chunk in enumerate(dom_chunks, start=1):
        try:
            response = chain.invoke(
                {"dom_content": chunk, "parse_description": parse_description}
            )
            print(f"Parsed batch: {i} of {len(dom_chunks)}")
            parsed_results.append(response)

        except httpx.ConnectError as e :
            print(f"Connection error: {e}")
            parsed_results.append(f"Failed to connect during batch {i}")

    return "\n".join(parsed_results)