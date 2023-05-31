from flask import Flask, request, jsonify
import os
import tempfile
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from config import WHITE, GREEN, RESET_COLOR, model_name
from utils import format_user_question
from file_processing import clone_github_repo, load_and_index_files
from questions import ask_question, QuestionContext

app = Flask(__name__)

OPENAI_API_KEY = "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"

@app.route('/ask', methods=['POST'])
def ask_question_api():
    try:
        data = request.get_json()
        user_question = data['question']
        
        github_url = "https://github.com/cmooredev/RepoReader"
        repo_name = github_url.split("/")[-1]
        with tempfile.TemporaryDirectory() as local_path:
            if clone_github_repo(github_url, local_path):
                index, documents, file_type_counts, filenames = load_and_index_files(local_path)
                if index is None:
                    return jsonify({'answer': "No documents were found to index. Exiting."})
                
                llm = OpenAI(api_key=OPENAI_API_KEY, temperature=0.2)
                template = """
                Repo: {repo_name} ({github_url}) | Conv: {conversation_history} | Docs: {numbered_documents} | Q: {question} | FileCount: {file_type_counts} | FileNames: {filenames}

                Instr:
                1. Answer based on context/docs.
                2. Focus on repo/code.
                3. Consider:
                    a. Purpose/features - describe.
                    b. Functions/code - provide details/samples.
                    c. Setup/usage - give instructions.
                4. Unsure? Say "I am not sure".

                Answer:
                """
                prompt = PromptTemplate(
                    template=template,
                    input_variables=["repo_name", "github_url", "conversation_history", "question", "numbered_documents", "file_type_counts", "filenames"]
                )
                llm_chain = LLMChain(prompt=prompt, llm=llm)
                conversation_history = ""
                question_context = QuestionContext(index, documents, llm_chain, model_name, repo_name, github_url, conversation_history, file_type_counts, filenames)
                
                user_question = format_user_question(user_question)
                answer = ask_question(user_question, question_context)
                
                conversation_history += f"Question: {user_question}\nAnswer: {answer}\n"
                
                return jsonify({'answer': answer})
            else:
                return jsonify({'answer': "Failed to clone the repository."})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run()
