# aurawear_analysis/recommend/llm/client.py

import os
import json
from typing import Any, Dict, Optional
from openai import OpenAI


class LLMClient:
    def __init__(
        self,
        model: str = "gpt-4.1-mini",
        *,
        client: Optional[OpenAI] = None,
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.client = client or OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def json_call(self, *, instructions: str, input_text: str, schema: Dict[str, Any]):
        full_input = instructions
        if input_text:
            full_input += "\n\n" + input_text

        resp = self.client.responses.create(
            model=self.model,
            input=full_input,
            text={
                "format": {
                    "type": "json_schema",
                    "name": "aurawear_schema",
                    "schema": schema,   # ← 現在 schema 已經是純 object
                    "strict": True,
                }
            },
        )

        return json.loads(resp.output_text)
