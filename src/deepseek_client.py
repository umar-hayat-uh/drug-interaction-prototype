import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def get_drug_description(drug_name):
    prompt = (
        f"Provide a clear and concise medical description of '{drug_name}', "
        "including its main uses and common side effects. Keep it short and simple."
    )

    try:
        response = client.chat.completions.create(
            model="openai/gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=400,
            extra_headers={
                "HTTP-Referer": "http://localhost:8501",
                "X-Title": "Drug Tools Suite"
            }
        )
        # Fix: access .content attribute instead of subscripting
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"
