# ChatWithPdf

## RUN COMMANDS :
- python run app/main.py
  (Use FastApi in swagger UI : 127.0.0.1:8000)

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
