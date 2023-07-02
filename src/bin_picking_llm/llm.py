"""Module to access ChatGPT."""

import json
import os
from typing import Callable, Dict, List

import openai


openai.api_key = os.getenv("OPENAI_API_KEY")


class ChatGPT:
    def __init__(
        self,
        functions: List,
        function_map: Dict[str, Callable],
        model: str = "gpt-3.5-turbo-0613",
    ):
        self._functions = functions
        self._function_map = function_map
        self._model = model
        self._messages = []

    def _create_message(self, prompt):
        return {"role": "user", "content": prompt}

    def _create_function_response_message(
        self,
        function_name,
        function_response,
    ):
        return {
            "role": "function",
            "name": function_name,
            "content": json.dumps(function_response),
        }

    def _call_function(self, function_name, arguments):
        fn = self._function_map.get(function_name)
        return fn(**arguments)

    def _process_message(self, message):
        self._messages.append(message)
        response = openai.ChatCompletion.create(
            model=self._model,
            messages=self._messages,
            functions=self._functions,
        )
        print(json.dumps(response))
        message = response.choices[0].message
        self._messages.append(message)

        if message.get("function_call"):
            # Call function if message contains "function_call"
            function_name = message["function_call"]["name"]
            arguments = json.loads(message["function_call"]["arguments"])
            function_response = self._call_function(function_name, arguments)

            # Create function response message, and process message recursively
            message = self._create_function_response_message(
                function_name,
                function_response,
            )
            return self._process_message(message)

        return message.get("content")

    def chat(self, prompt):
        message = self._create_message(prompt)
        return self._process_message(message)
