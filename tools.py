# File: tools.py
from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
import os # Importar el módulo os

def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    # Definir la carpeta de destino
    research_folder = "research"

    # Crear la carpeta si no existe
    if not os.path.exists(research_folder):
        os.makedirs(research_folder)

    # Construir la ruta completa del archivo
    full_filepath = os.path.join(research_folder, filename)

    # Guardar el archivo en la ruta completa
    with open(full_filepath, "a", encoding="utf-8") as f:
        f.write(formatted_text)

        return f"Data successfully saved to {full_filepath}" # Actualizar el mensaje de éxito

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

search = DuckDuckGoSearchRun()
search_tool = Tool(
    name="search",
    func=search.run,
    description="Search the web for information",
)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
