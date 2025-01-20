# ChatWithPdf

## RUN COMMANDS :
- python run app/main.py
  (Use FastApi in swagger UI : 127.0.0.1:8000)

## TESTRUN - FLOW :
- Upload PDF file :
    - Incase of Duplicate Document , Record Manager detects and pushes a already uploaded document response
    - Else, the list of chunked doucments will be pushed to vector store and respond with pdf uploaded successfully message.
- Chat with pdf :
    - implemented a QA retrieval chain that combines retrieving documents from a vector store retriever and responds to the QA as response
    - citations will be provided from the meta data of documents retrieved
- Chat wth pdf (latest):
 - latest endpoint , provided with multi query retrieval features that provides nion of unique documents,
    from multiple queries and a LLM chain to respond to the QA and documents used for that respinse directly from LLM
 - this LLM response is parsed to a pydantic output parser and a custom validator to provide only unique citations
 - what fixed ? customizable prompt based chains, pydantic output parser
   - why fixed ? previous endpoint couldn't provide which unique documents have been used 
     - ie. when asking about vector -> previous system cited vector/vector algebra documents , which was retrieved from documents
     - in this system it only cites the used documents as it is provided directly from LLM response which then validated by a Pydantic Model.


# next-version-update features :
## Code Fixes :
- Configure run attributes from .env file
    - LLM models
    - Embedding Models
    - Vector Store
    - Data Sink Paths (Optional, if stored in Docker Volumes)

- Update Helper methods to be run from .env Configurations -> (not fixed yet in code)

## Design Features
- Query Segregation :
    - User Query -> DB Query, LLM Query
    - DB Query specific to retrieve similar documents from vector store
    - LLM query specific to prompt LLM's to provide apt answers.

- Reconcilating Record Storage -> [Consistency across Indexing and VectorDatabase]
    - Verify records across index Record Manager are consistenly added to Vector Databases.

- Containerizing File Storage
    - Select Host for Data Sinks -> File System of System / Containerized File Hosting (eg: Volumes in Docker Containers)

## App features
- Support multiple file types .csv/.pdf etc ...
- Query rooting to multiple types of queries 
