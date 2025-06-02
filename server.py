from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
load_dotenv

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
)

prompt_template = """[SYSTEM PROMPT]
You are a professional summarization assistant. Follow these steps:
1. Correct any grammatical errors in the transcript
2. Remove filler words and repetitions
3. Generate a concise summary preserving key points
4. Maintain original intent and formal tone

Transcript: {text}

Summary:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
chain = LLMChain(llm=llm, prompt=prompt)


@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text', '')

    if not text:
        return jsonify({'summary': 'No text provided'}), 400

    try:
        result = chain.invoke({'text': text})
        return jsonify({'summary': result['text']})
    except Exception as e:
        return jsonify({'summary': f'Error: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
