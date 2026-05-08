"""Ground-truth QA pairs used by the RAG and RAGAS lab steps."""

QA_PAIRS = [
    {
        "question": "What are the three main types of machine learning?",
        "reference": "The three main types of machine learning are supervised learning, unsupervised learning, and reinforcement learning.",
    },
    {
        "question": "What is overfitting in machine learning?",
        "reference": "Overfitting occurs when a model learns the training data too well, including noise, leading to poor generalization on new data.",
    },
    {
        "question": "Explain the bias-variance tradeoff.",
        "reference": "High bias means underfitting; high variance means overfitting. The optimal model balances both to minimize total error.",
    },
    {
        "question": "How does regularization prevent overfitting?",
        "reference": "L1 and L2 regularization add penalty terms to the loss function to discourage overly complex models.",
    },
    {
        "question": "What is cross-validation?",
        "reference": "Cross-validation splits data into multiple folds to estimate model performance more reliably.",
    },
    {
        "question": "What is backpropagation?",
        "reference": "Backpropagation computes gradients of the loss with respect to each weight by applying the chain rule of calculus.",
    },
    {
        "question": "What are Convolutional Neural Networks primarily used for?",
        "reference": "CNNs are specialized for processing grid-like data such as images.",
    },
    {
        "question": "How do LSTM networks address the vanishing gradient problem?",
        "reference": "LSTMs use gating mechanisms that control information flow through the network.",
    },
    {
        "question": "What activation functions are commonly used in neural networks?",
        "reference": "Common activation functions include ReLU, sigmoid, and tanh.",
    },
    {
        "question": "What is the role of pooling layers in CNNs?",
        "reference": "Pooling layers reduce spatial dimensions, decreasing computational complexity while retaining important features.",
    },
    {
        "question": "What is the transformer architecture?",
        "reference": "The transformer uses self-attention mechanisms and processes entire sequences in parallel, introduced in 'Attention Is All You Need' (2017).",
    },
    {
        "question": "What are word embeddings?",
        "reference": "Word embeddings represent words as dense vectors where semantically similar words are geometrically close.",
    },
    {
        "question": "What is transfer learning in NLP?",
        "reference": "Pre-training large models on massive corpora then fine-tuning on downstream tasks.",
    },
    {
        "question": "How does BERT handle language understanding?",
        "reference": "BERT uses bidirectional transformer training with masked language modeling and next sentence prediction.",
    },
    {
        "question": "What is self-attention in transformers?",
        "reference": "Self-attention allows models to weigh the importance of different words relative to each other in a sequence.",
    },
    {
        "question": "What is GPT and how is it trained?",
        "reference": "GPT uses autoregressive training to predict the next token given previous tokens, trained on massive text datasets.",
    },
    {
        "question": "What is instruction tuning?",
        "reference": "Fine-tuning pre-trained LLMs on instruction-following datasets to improve alignment with human intent.",
    },
    {
        "question": "What is RLHF?",
        "reference": "Reinforcement Learning from Human Feedback uses human preferences to align LLMs to be helpful, harmless, and honest.",
    },
    {
        "question": "What is chain-of-thought prompting?",
        "reference": "Chain-of-thought prompting encourages LLMs to show reasoning step by step, improving performance on complex tasks.",
    },
    {
        "question": "What is the context length of GPT-4?",
        "reference": "GPT-4 supports up to 128K tokens of context.",
    },
    {
        "question": "What is Retrieval-Augmented Generation?",
        "reference": "RAG combines generative LLMs with retrieval from external knowledge bases to produce grounded, up-to-date answers.",
    },
    {
        "question": "What are the main components of a RAG pipeline?",
        "reference": "A retriever that searches a document store and a generator (LLM) that produces answers from the query and retrieved passages.",
    },
    {
        "question": "What is dense retrieval?",
        "reference": "Dense retrieval uses neural embeddings to encode queries and documents, determining relevance by cosine similarity.",
    },
    {
        "question": "Why is chunking strategy important in RAG?",
        "reference": "Chunking affects retrieval precision and context window fit; options include fixed-size, semantic, and hierarchical chunking.",
    },
    {
        "question": "What advanced RAG techniques exist beyond basic retrieval?",
        "reference": "Re-ranking, query expansion, HyDE, and iterative retrieval for multi-hop questions.",
    },
    {
        "question": "What are vector databases used for?",
        "reference": "Storing and querying high-dimensional vector embeddings for fast similarity search.",
    },
    {
        "question": "What is FAISS?",
        "reference": "FAISS is a library for efficient similarity search supporting exact and approximate nearest neighbor algorithms.",
    },
    {
        "question": "How do text embeddings capture semantic meaning?",
        "reference": "Text embeddings convert text into numerical vectors where semantically similar texts produce geometrically close vectors.",
    },
    {
        "question": "What is HNSW?",
        "reference": "HNSW builds a hierarchical graph for logarithmic-complexity approximate nearest neighbor search.",
    },
    {
        "question": "What is hybrid search in vector databases?",
        "reference": "Hybrid search combines dense vector search with sparse keyword search (BM25) using Reciprocal Rank Fusion.",
    },
    {
        "question": "What is LangChain?",
        "reference": "LangChain is an open-source framework for building LLM applications with abstractions for chains, agents, and data connections.",
    },
    {
        "question": "What is LangChain Expression Language (LCEL)?",
        "reference": "LCEL is a declarative way to compose chains using the pipe operator, supporting streaming, async, and batching.",
    },
    {
        "question": "What is LangGraph?",
        "reference": "LangGraph extends LangChain for stateful multi-actor applications as directed graphs, supporting cycles.",
    },
    {
        "question": "What memory types does LangChain support?",
        "reference": "ConversationBufferMemory, ConversationSummaryMemory, ConversationWindowMemory, and vector store memory.",
    },
    {
        "question": "What are LangChain retrievers?",
        "reference": "Retrievers fetch relevant documents from a data source given a query, supporting vector stores and BM25.",
    },
    {
        "question": "What is LangSmith?",
        "reference": "LangSmith is a platform for debugging, testing, evaluating, and monitoring LLM applications through automatic tracing.",
    },
    {
        "question": "What information do LangSmith traces capture?",
        "reference": "Inputs, outputs, latency, token usage, and errors for every component in a chain.",
    },
    {
        "question": "What is the LangSmith Prompt Hub?",
        "reference": "A repository for storing, versioning, and sharing prompt templates, supporting A/B testing and rollback.",
    },
    {
        "question": "How does LangSmith help monitor production LLM applications?",
        "reference": "LangSmith provides latency percentiles, error rates, token costs, and feedback annotations.",
    },
    {
        "question": "What are LangSmith datasets used for?",
        "reference": "Systematic evaluation using example inputs and expected outputs to compare model versions or prompt changes.",
    },
    {
        "question": "What is RAGAS?",
        "reference": "RAGAS is an open-source framework for evaluating RAG pipelines using LLM-based reference-free metrics.",
    },
    {
        "question": "How does RAGAS compute faithfulness?",
        "reference": "By extracting claims from the answer and checking whether each claim can be inferred from the retrieved context.",
    },
    {
        "question": "What is answer relevancy in RAGAS?",
        "reference": "Measures how well the answer addresses the original question by generating synthetic questions from the answer.",
    },
    {
        "question": "What is context recall in RAGAS?",
        "reference": "Measures how well the retrieved context covers the information needed, evaluated against a ground truth reference.",
    },
    {
        "question": "What inputs does RAGAS evaluation require?",
        "reference": "User queries, generated answers, retrieved contexts (list of passages), and optionally reference answers.",
    },
    {
        "question": "What is Guardrails AI?",
        "reference": "An open-source framework for adding validation and safety checks to LLM outputs with configurable on-fail actions.",
    },
    {
        "question": "What is PII and why is it important to detect in LLM responses?",
        "reference": "PII is Personally Identifiable Information; exposing it can violate GDPR and HIPAA privacy regulations.",
    },
    {
        "question": "What does structured output validation ensure?",
        "reference": "That LLM responses conform to expected schemas such as JSON, detecting and fixing invalid formats.",
    },
    {
        "question": "What is Constitutional AI?",
        "reference": "A technique where a model is given principles to follow and uses self-critique to improve its responses.",
    },
    {
        "question": "What are common AI safety concerns with LLMs?",
        "reference": "Hallucination, toxicity, bias, PII leakage, and jailbreaking attacks.",
    },
]

SAMPLE_QUESTIONS = [pair["question"] for pair in QA_PAIRS]
