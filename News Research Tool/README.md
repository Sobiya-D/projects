# **News Research Tool**

![News Research Tool](bot.jpg)

## **Description**
This tool is designed to enhance the retrieval of critical information from news articles, with a focus on the financial and stock market sectors. Users can input article URLs to quickly access detailed insights and answers, powered by advanced content processing and similarity search technologies.

## **Features**

- **URL Handling**: Easily load URLs or upload text files containing multiple URLs to extract article content.
- **Content Processing**: Utilize LangChain's UnstructuredURL Loader to process and structure the extracted article content efficiently.
- **Advanced Embedding & Search**: Create embedding vectors using OpenAI's embeddings and employ FAISS, a powerful similarity search library, for rapid and precise information retrieval.
- **Interactive Querying**: Engage with the tool by asking questions, and receive accurate answers, complete with source URLs, leveraging LLMs like ChatGPT.

## **Project Structure**

- **`main.py`**: The primary script for running the Streamlit application.
- **`requirements.txt`**: Contains the list of required Python packages.
- **`.env`**: Configuration file for storing the OpenAI API key.