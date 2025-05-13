# AI Research Agent 🤖

This project is an application that uses Langchain and other tools to perform research and save the results. The user interface is built with Streamlit.

## Files and Structure
- app.py: Contains the main logic for the Streamlit application and interaction with the research agent. 
- tools.py: Likely contains the definitions of the tools used by the agent (like the tool to save text to a file). 
- requirements.txt: Lists the necessary Python dependencies to run the project. research/: Folder to store research results. 
- research/research_output.txt: File where the research results are saved.

## Setup and Installation
Clone this repository:
```bash
git clone https://github.com/davila7/research_agent 
cd research_agent
```

## Create a virtual environment (recommended):
```bash
python -m venv .venv source .venv/bin/activate # On Windows use .venv\Scripts\activate
```

## Install dependencies:
```bash
pip install -r requirements.txt
```

## Configure your API keys (if necessary). 
This project uses langchain-openai, langchain-anthropic, langchain_mistralai, so you might need to set environment variables like OPENAI_API_KEY, ANTHROPIC_API_KEY, MISTRAL_API_KEY, etc. Consult the Langchain documentation for more details.
Execution

To run the Streamlit application, use the following command:
```bash
streamlit run app.py
```
This will open the application in your web browser.

## License
This project is licensed under the MIT 