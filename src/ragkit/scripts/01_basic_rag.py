"""Basic RAG pipeline."""

import argparse

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from ragkit.utils import format_docs, get_llm, get_vectorstore, load_web, split_docs

URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"


def build_chain():
    docs = load_web(URL)
    splits = split_docs(docs)
    vectorstore = get_vectorstore(splits, name="01_basic")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_template(
        """Answer the question based only on the context:

{context}

Question: {question}"""
    )

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return chain


def main():
    parser = argparse.ArgumentParser(description="Basic RAG")
    parser.add_argument("--question", required=True)
    args = parser.parse_args()

    chain = build_chain()
    print(chain.invoke(args.question))


if __name__ == "__main__":
    main()
