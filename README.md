Pet Project: RAG-Confluence App
Goal: Simplify interaction with Confluence using Retrieval-Augmented Generation (RAG)
Stack: Chainlit, RAG, Flyway, Deepseek
Workflow:

Data loading:

- A script extracts specified Confluence spaces
- Splits content into chunks
- Converts chunks into vectors using an embedding model
- Stores vectors in a PostgreSQL vector database


Query processing:

- The user's query is converted into a vector
- Nearest chunks are retrieved based on cosine similarity
- The query + chunks are passed to an LLM for answer generation
