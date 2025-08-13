from openai import OpenAI  # works with Perplexity API setup
import os  

client = OpenAI(api_key="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", base_url="https://api.perplexity.ai")  # if using Perplexity API


def generate_narrative(section, data_points, stakeholder_type="general"):
    messages = [
        {"role": "system", "content": "You are a public health data analyst writing clear, concise, human-centred explanations for report sections."},
        {"role": "user", "content": f"Write the {section} for a {stakeholder_type} audience, using these findings: {data_points}"}
    ]
    response = client.chat.completions.create(
        model="sonar-pro",        # Or any Perplexity-supported model you want
        messages=messages,
        max_tokens=250,
        temperature=0.5,
    )
    narrative = response.choices[0].message.content.strip()
    return narrative
