# Real Estate Assistant

A powerful multi-agent AI system that helps users with property issues and tenancy questions.

## Overview

The Real Estate Assistant is a Streamlit web application that uses multiple specialized AI agents to address different types of real estate queries. It leverages LangGraph for orchestration and Groq's large language models for natural language understanding and generation, with special vision capabilities for analyzing property images.

## Features

- **Multi-agent architecture** - Routes queries to specialized agents
- **Image analysis** - Upload photos of property issues for AI visual inspection
- **Tenancy support** - Get answers about rental agreements, tenant rights, and more
- **Interactive chat interface** - Simple, user-friendly communication

## Tech Stack

- **Streamlit**: Frontend web application
- **LangGraph**: Agent orchestration framework
- **Groq AI**: LLM API for text processing
- **Llama-4-Scout**: Vision-capable model for image analysis
- **Python 3.9+**: Core programming language

## Getting Started

### Prerequisites

- Python 3.9+
- Groq API key

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/real-estate-assistant.git
   cd real-estate-assistant
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:

   ```bash
   export GROQ_API_KEY="your_groq_api_key"
   ```

   Or create a `.streamlit/secrets.toml` file:

   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   ```

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Type your property or tenancy question in the chat input
2. For property issues, upload an image of the problem (optional)
3. The system will route your question to the appropriate specialist agent
4. Receive expert advice tailored to your specific query

## Example Queries

- "There's a water stain on my ceiling. What should I do?" (upload an image for best results)
- "Is my landlord responsible for fixing the refrigerator?"
- "What should I include in a lease agreement for my rental property?"
- "How do I address mold growing in the bathroom?"

## License

[MIT License](LICENSE)

## Acknowledgements

- [Streamlit](https://streamlit.io/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Groq](https://groq.com/)
- [Meta AI](https://ai.meta.com/) for the Llama model family
