from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import hashlib
import time
from typing import Literal, Optional
from openai import OpenAI
from openai.types.chat import ChatCompletion

from llm_guard import scan_output, scan_prompt
from llm_guard.input_scanners import Anonymize, PromptInjection, TokenLimit, Toxicity
from llm_guard.output_scanners import Deanonymize, NoRefusal, Relevance, Sensitive
from llm_guard.vault import Vault

vault = Vault()
input_scanners = [Anonymize(vault), Toxicity(), TokenLimit(), PromptInjection()]
output_scanners = [Deanonymize(vault), NoRefusal(), Relevance(), Sensitive()]

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "gpt-4o"
    messages: list[ChatMessage]
    # What 4o uses by default on openai playground
    max_tokens: Optional[int] = 2048
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0


class LLMServer:
    def __init__(self):
        self.client = OpenAI()

    def __call__(self, temperature, top_p, max_tokens, messages, model) -> ChatCompletion:
        return self.client.chat.completions.create(
            messages=messages,
            model=model,
            top_p=top_p,
            max_tokens=max_tokens,
            temperature=temperature,
        )


llm_server = LLMServer()

app = FastAPI(title="OpenAI-compatible API")


@app.post("/v1/chat/completions")
def chat_completions(
    request: ChatCompletionRequest
):
    new_messages: list[ChatMessage] = []
    for index, message in enumerate(request.messages):
        message_content = message.content
        input_prompt, input_results_valid, input_results_score = scan_prompt(input_scanners, message_content)
        if any(input_results_valid.values()) is False:
            print(f"Prompt {message_content} is not valid, scores: {input_results_score}")

            return {
                "id": hashlib.sha256(str(request.messages).encode()).hexdigest(),
                "object": "chat.completion",
                "created": time.time(),
                "model": request.model,
                "choices": [
                    {"message": ChatMessage(role="assistant", content=f"LLM Guard detected badness on input {message_content} {input_results_score}")}
                ],
            }
        new_messages.append(
            ChatMessage(role=request.messages[index].role,
                        content=input_prompt)
                        )

    response_content = llm_server(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        messages=new_messages,
        model=request.model
    )

    response_text = response_content.choices[0].message.content
    outputs_sanitized_response, output_results_valid, output_results_score = scan_output(
        output_scanners, input_prompt, response_text
    )
    if any(output_results_valid.values()) is False:
        print(f"Output {response_text} is not valid, scores: {output_results_score}")

        return {
                "id": hashlib.sha256(str(request.messages).encode()).hexdigest(),
                "object": "chat.completion",
                "created": time.time(),
                "model": request.model,
                "choices": [
                    {"message": ChatMessage(role="assistant", content=f"LLM Guard detected badness on output {response_text} {output_results_score}")}
                ],
            }

    return {
        "id": hashlib.sha256(str(request.messages).encode()).hexdigest(),
        "object": "chat.completion",
        "created": time.time(),
        "model": request.model,
        "choices": [
            {"message": ChatMessage(role="assistant", content=outputs_sanitized_response)}
        ],
    }


if __name__ == "__main__":
    import uvicorn

    print("Running...")
    uvicorn.run(app)
