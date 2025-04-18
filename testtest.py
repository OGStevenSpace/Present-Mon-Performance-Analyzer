import re
import asyncio
from ollama import chat


class AsyncGeneratorWrapper:
    def __init__(self, generator):
        self.generator = generator

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.generator)
        except StopIteration:
            raise StopAsyncIteration


async def request(input_content, system_prompt, deep_think=True):
    response_text = ''

    async for chunk in AsyncGeneratorWrapper(chat(
        model='deepseek-r1',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': input_content}
        ],
        stream=True,
    )):
        content = chunk['message']['content']
        response_text += content
    think_texts = ' '.join(re.findall(r'<think>(.*?)</think>', response_text, flags=re.DOTALL)).strip()
    clean_response = ' '.join(re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).split())
    print(clean_response)
    return clean_response if not deep_think else (think_texts, clean_response)