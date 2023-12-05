# PDFChat (CET Advisement Bot Test Tool)

### A basic PDF querying program using GPT-3.5 to retrieve information.

## Instructions to run locally
Follow these steps to get started:

### Prerequisites
- Python 3.11 or higher
- Git

### Installation
Clone the repository:

`git clone https://github.com/sahsan2018/PDFChat.git`


Navigate to the project directory:

`cd PDFChat`


Create a virtual environment:
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

Install the required dependencies in the virtual environment:

`pip install -r requirements.txt`

Store your OpenAI API key in a `.env` file:

`echo "OPENAI_API_KEY=YOUR_OPENAI_API_KEY" >> .env`

### Running the application
Select one of the sample PDF files to process like so in `vector_store.py` (rename the file path to reflect the desired document):

```bash
# Load PDF
loaders = [
    PyPDFLoader("docs/Case4.pdf")
]
```

Run `vector_store.py` and wait for the vector database to be created

Finally, run `chat.py` to see how the default sample question works, or create your own in the following lines of `chat.py`:

```bash
# Run chain
question = "What requirements still need to be met?"
qa_chain = RetrievalQA.from_chain_type(llm,
                                       retriever=compression_retriever,
                                       return_source_documents=True,
                                       chain_type_kwargs={"prompt": QA_CHAIN_PROMPT})
```

And that's all for this test application!