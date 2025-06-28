# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Some LLM inference interface."""

import abc
import base64
import io
import os
import pdb
import time
from typing import Any, Optional
import google.generativeai as genai
from google.generativeai import types
from google.generativeai.types import answer_types
from google.generativeai.types import content_types
from google.generativeai.types import generation_types
from google.generativeai.types import safety_types
import numpy as np
from PIL import Image
import requests
from openai import AzureOpenAI, OpenAI
import torch
from transformers import MllamaForConditionalGeneration, AutoProcessor
from typing import Optional, Union, List

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from llama_cpp import Llama
from peft import PeftModel
from vllm import SamplingParams
from vllm.lora.request import LoRARequest

from android_world.agents.reward_model import LlamaRewardModel, load_lora_model_from_dir

from azure.identity import AzureCliCredential, get_bearer_token_provider

trapi_deployment_list = [
    'gpt-4_0125-Preview',
    'gpt-4o_2024-11-20',
    'gpt-4o-mini_2024-07-18',
    'deepseek-r1',
]

# env variables for calling third party APIs

# use trapi LLM service, which is keyless, requires Azure CLI login
TRAPI_ENDPOINT = os.environ.get('TRAPI_ENDPOINT', None)
TRAPI_MODEL_NAME = os.environ.get('TRAPI_MODEL_NAME', None)
TRAPI_API_VERSION = os.environ.get('TRAPI_API_VERSION', None)

# use Gemini GCP service, which requires API key
GCP_API_KEY = os.environ.get('GCP_API_KEY', None)


# use openai compatible APIs, including OPENAI, Qwen and DeepSeek
OPENAI_ENDPOINT = os.environ.get('OPENAI_ENDPOINT', None)
OPENAI_MODEL_NAME = os.environ.get('OPENAI_MODEL_NAME', None)
OPENAI_API_VERSION = os.environ.get('OPENAI_API_VERSION', None)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', None)


# use azure openai services
AZURE_OPENAI_API_KEY = os.environ.get('AZURE_OPENAI_API_KEY', None)
AZURE_OPENAI_MODEL_NAME = os.environ.get('AZURE_OPENAI_MODEL_NAME', None)
AZURE_OPENAI_API_VERSION = os.environ.get('AZURE_OPENAI_API_VERSION', None)
AZURE_OPENAI_ENDPOINT = os.environ.get('AZURE_OPENAI_ENDPOINT', None)

ERROR_CALLING_LLM = 'Error calling LLM'


def array_to_jpeg_bytes(image: np.ndarray) -> bytes:
    """Converts a numpy array into a byte string for a JPEG image."""
    image = Image.fromarray(image)
    return image_to_jpeg_bytes(image)


def image_to_jpeg_bytes(image: Image.Image) -> bytes:
    in_mem_file = io.BytesIO()
    image.save(in_mem_file, format='JPEG')
    # Reset file pointer to start
    in_mem_file.seek(0)
    img_bytes = in_mem_file.read()
    return img_bytes


class LlmWrapper(abc.ABC):
    """Abstract interface for (text only) LLM."""

    @abc.abstractmethod
    def predict(
        self,
        text_prompt: str,
    ) -> tuple[str, Optional[bool], Any]:
        """Calling multimodal LLM with a prompt and a list of images.

        Args:
          text_prompt: Text prompt.

        Returns:
          Text output, is_safe, and raw output.
        """


class MultimodalLlmWrapper(abc.ABC):
    """Abstract interface for Multimodal LLM."""

    @abc.abstractmethod
    def predict_mm(
        self, text_prompt: str, images: list[np.ndarray], max_token_len: int,
    ) -> tuple[str, Optional[bool], Any]:
        """Calling multimodal LLM with a prompt and a list of images.

        Args:
          text_prompt: Text prompt.
          images: List of images as numpy ndarray.

        Returns:
          Text output and raw output.
        """

    @abc.abstractmethod
    def predict_logits(
        self, text_prompt: str, images: list[np.ndarray], max_token_len: int,
    ) -> tuple[list, Optional[bool], Any]:
        """Calling multimodal LLM with a prompt and a list of images.

        Args:
          text_prompt: Text prompt.
          images: List of images as numpy ndarray.

        Returns:
          list of logits and raw output.
        """

    @abc.abstractmethod
    def predict_logits_batch(
        self, text_prompt: list[str], images: list[np.ndarray], max_token_len: int,
    ) -> tuple[list, Optional[bool], Any]:
        """Calling multimodal LLM with a prompt and a list of images.

        Args:
          text_prompt: Text prompt.
          images: List of images as numpy ndarray.

        Returns:
          list of logits and raw output.
        """

    @abc.abstractmethod
    def predict_scores_batch(
        self, text_prompt: list[str], images: list[np.ndarray], max_token_len: int,
    ) -> tuple[list, Optional[bool], Any]:
        """Calling multimodal LLM with a prompt and a list of images.

        Args:
          text_prompt: Text prompt.
          images: List of images as numpy ndarray.

        Returns:
          list of logits and raw output.
        """


SAFETY_SETTINGS_BLOCK_NONE = {
    types.HarmCategory.HARM_CATEGORY_HARASSMENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
    types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: (
        types.HarmBlockThreshold.BLOCK_NONE
    ),
}


class GeminiGcpWrapper(LlmWrapper, MultimodalLlmWrapper):
    """Gemini GCP interface."""

    def __init__(
        self,
        model_name: str | None = None,
        max_retry: int = 100,
        temperature: float = 0.0,
        top_p: float = 0.95,
        enable_safety_checks: bool = False,
    ):
        if 'GCP_API_KEY' not in os.environ:
            raise RuntimeError('GCP API key not set.')

        # pdb.set_trace()
        genai.configure(api_key=os.environ['GCP_API_KEY'],)
        # transport="rest",
        # client_options={"api_endpoint": "https://api.openai-proxy.org/google",})
        self.generation_config = generation_types.GenerationConfig(
            temperature=temperature, top_p=top_p,
            # max_output_tokens=1000
        )
        self.model_name = model_name
        self.llm = genai.GenerativeModel(
            model_name,
            safety_settings=None
            if enable_safety_checks
            else SAFETY_SETTINGS_BLOCK_NONE,
            generation_config=generation_types.GenerationConfig(
                temperature=temperature, top_p=top_p,
                # max_output_tokens=1000
            ),
        )
        if max_retry <= 0:
            max_retry = 100
            print('Max_retry must be positive. Reset it to 100')
        self.max_retry = min(max_retry, 100)

    def predict(
        self,
        text_prompt: str,
        enable_safety_checks: bool = False,
        generation_config: generation_types.GenerationConfigType | None = None,
    ) -> tuple[str, Optional[bool], Any]:
        return self.predict_mm(
            text_prompt, [], enable_safety_checks, self.generation_config
        )

    def is_safe(self, raw_response):
        try:
            return (
                raw_response.candidates[0].finish_reason
                != answer_types.FinishReason.SAFETY
            )
        except Exception:  # pylint: disable=broad-exception-caught
            #  Assume safe if the response is None or doesn't have candidates.
            return True

    def predict_mm(
        self,
        text_prompt: str,
        images: list[np.ndarray],
        enable_safety_checks: bool = False,
        generation_config: generation_types.GenerationConfigType | None = None,
    ) -> tuple[str, Optional[bool], Any]:
        counter = self.max_retry
        retry_delay = 5.0
        output = None
        while counter > 0:
            try:
                output = self.llm.generate_content(
                    [text_prompt] +
                    [Image.fromarray(image) for image in images],
                    safety_settings=None
                    if enable_safety_checks
                    else SAFETY_SETTINGS_BLOCK_NONE,
                    generation_config=self.generation_config,
                )
                return output.text, True, output
            except Exception as e:
                counter -= 1
                print(
                    f'Error calling LLM, will retry in {retry_delay} seconds')
                print(e)
                if counter > 0:
                    # Expo backoff
                    time.sleep(retry_delay)
                    # retry_delay *= 2

        if (output is not None) and (not self.is_safe(output)):
            return ERROR_CALLING_LLM, False, output
        return ERROR_CALLING_LLM, None, None

    def generate(
        self,
        contents: (
            content_types.ContentsType | list[str | np.ndarray | Image.Image]
        ),
        safety_settings: safety_types.SafetySettingOptions | None = None,
        generation_config: generation_types.GenerationConfigType | None = None,
    ) -> tuple[str, Any]:
        """Exposes the generate_content API.

        Args:
          contents: The input to the LLM.
          safety_settings: Safety settings.
          generation_config: Generation config.

        Returns:
          The output text and the raw response.
        Raises:
          RuntimeError:
        """
        counter = self.max_retry
        retry_delay = 5.0
        response = None
        if isinstance(contents, list):
            contents = self.convert_content(contents)
        while counter > 0:
            try:
                response = self.llm.generate_content(
                    contents=contents,
                    safety_settings=safety_settings,
                    generation_config=generation_config,
                )
                return response.text, response
            except Exception as e:  # pylint: disable=broad-exception-caught
                counter -= 1
                print(
                    f'Error calling LLM, will retry in {retry_delay} seconds')
                print(e)
                if counter > 0:
                    # Expo backoff
                    time.sleep(retry_delay)
                    # retry_delay *= 2
        raise RuntimeError(f'Error calling LLM. {response}.')

    def convert_content(
        self,
        contents: list[str | np.ndarray | Image.Image],
    ) -> content_types.ContentsType:
        """Converts a list of contents to a ContentsType."""
        converted = []
        for item in contents:
            if isinstance(item, str):
                converted.append(item)
            elif isinstance(item, np.ndarray):
                converted.append(Image.fromarray(item))
            elif isinstance(item, Image.Image):
                converted.append(item)
        return converted

    def predict_logits(
        self, text_prompt: str, images: list[np.ndarray], max_token_len: int,
    ) -> tuple[list, Optional[bool], Any]:
        return

    def predict_logits_batch(
        self, text_prompt: list[str], images: list[np.ndarray], max_token_len: int,
    ) -> tuple[list, Optional[bool], Any]:
        return

    def predict_scores_batch(
        self, text_prompt: list[str], images: list[np.ndarray], max_token_len: int,
    ) -> tuple[list, Optional[bool], Any]:
        return


# def store_screen(state, step_idx, save_dir):
#     # pdb.set_trace()
#     save_path = os.path.join(save_dir, f"screen_shot/")
#     if not os.path.exists(save_path):
#         os.makedirs(save_path)

#     pixels = state['raw_screenshot']
#     img = Image.fromarray(np.uint8(pixels))
#     img.save(save_path + f"step{step_idx}.jpg", 'JPEG')

#     pixels_ann = state['before_screenshot_with_som']
#     img_ann = Image.fromarray(np.uint8(pixels_ann))
#     img_ann.save(save_path + f"step{step_idx}_ann.jpg", 'JPEG')
#     return


class Gpt4Wrapper(LlmWrapper, MultimodalLlmWrapper):
    """OpenAI GPT4 wrapper.

    Attributes:
      openai_api_key: The class gets the OpenAI api key either explicitly, or
        through env variable in which case just leave this empty.
      max_retry: Max number of retries when some error happens.
      temperature: The temperature parameter in LLM to control result stability.
      model: GPT model to use based on if it is multimodal.
    """

    RETRY_WAITING_SECONDS = 1

    def __init__(
        self,
        model_name: str,
        service_name: str,
        max_retry: int = 100,
        temperature: float = 0.0,
    ):
        if max_retry <= 0:
            max_retry = 3
            print('Max_retry must be positive. Reset it to 3')
        # self.max_retry = min(max_retry, 5)
        self.max_retry = max_retry
        self.temperature = temperature
        self.model_name = model_name
        self.service_name = service_name

    @classmethod
    def encode_image(cls, image: np.ndarray) -> str:
        return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

    def predict(
        self,
        text_prompt: str,
        measure_time=False,
    ) -> tuple[str, Optional[bool], Any]:
        return self.predict_mm(text_prompt, [], measure_time)

    def predict_mm(
        self, text_prompt: str, images: list[np.ndarray], measure_time=False
    ) -> tuple[str, Optional[bool], Any]:

        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                ],
            }],
            "max_tokens": 200,
        }

        # Gpt-4v supports multiple images, just need to insert them in the content
        # list.

        if self.model_name in ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4-turbo-v']:
            for image in images:
                payload["messages"][0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.encode_image(image)}"
                    },
                })

        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS

        if self.service_name == 'openai':
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_ENDPOINT)
            self.deployment = OPENAI_MODEL_NAME

        elif self.service_name == 'trapi':
            scope = "api://trapi/.default"
            credential = get_bearer_token_provider(AzureCliCredential(), scope)
            client = AzureOpenAI(
                azure_endpoint=TRAPI_ENDPOINT,
                azure_ad_token_provider=credential,
                api_version=TRAPI_API_VERSION,
            )
            self.deployment = TRAPI_MODEL_NAME

        elif self.service_name == 'azure_openai':
            client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                azure_deployment=AZURE_OPENAI_MODEL_NAME
            )
            self.deployment = AZURE_OPENAI_MODEL_NAME

        else:
            raise ValueError('Unknown service name: %s' % self.service_name)

        while counter > 0:
            try:
                step_start_time = time.perf_counter()
                response = client.chat.completions.create(
                    messages=payload["messages"],
                    model=self.deployment,
                    temperature=self.temperature,
                    max_tokens=1000,
                )
                step_end_time = time.perf_counter()
                step_time = step_end_time - step_start_time
                if response and hasattr(response, 'choices'):
                    if measure_time:
                        return (
                            response.choices[0].message.content,
                            step_time,
                            response,
                        )
                    else:
                        return (
                            response.choices[0].message.content,
                            None,
                            response,
                        )
                print(
                    'Error calling OpenAI API with error message: '
                    + response["error"]["message"]
                )
                time.sleep(wait_seconds)
                # wait_seconds *= 2
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Want to catch all exceptions happened during LLM calls.
                time.sleep(wait_seconds)
                # wait_seconds *= 2
                counter -= 1
                print('Error calling LLM, will retry soon...')
                print(e)
        return ERROR_CALLING_LLM, None, None

    def predict_logits(self, text_prompt: str, images: list[np.ndarray]
                       ) -> tuple[str, Optional[bool], Any]:
        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                ],
            }],
            "max_tokens": 200,
        }

        if self.model_name in ['gpt-4o', 'gpt-4o-mini', 'gpt-4-turbo', 'gpt-4-turbo-v']:
            for image in images:
                payload["messages"][0]["content"].append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{self.encode_image(image)}"
                    },
                })

        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS
        if self.service_name == 'openai':
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_ENDPOINT)
            self.deployment = OPENAI_MODEL_NAME

        elif self.service_name == 'trapi':
            scope = "api://trapi/.default"
            credential = get_bearer_token_provider(AzureCliCredential(), scope)
            client = AzureOpenAI(
                azure_endpoint=TRAPI_ENDPOINT,
                azure_ad_token_provider=credential,
                api_version=TRAPI_API_VERSION,
            )
            self.deployment = TRAPI_MODEL_NAME

        elif self.service_name == 'azure_openai':
            client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                azure_deployment=AZURE_OPENAI_MODEL_NAME
            )
            self.deployment = AZURE_OPENAI_MODEL_NAME

        else:
            raise ValueError('Unknown service name: %s' % self.service_name)

        while counter > 0:
            try:
                response = client.chat.completions.create(
                    messages=payload["messages"],
                    model=self.deployment,
                    temperature=self.temperature,
                    max_tokens=1,
                    logprobs=True,
                )
                # pdb.set_trace()
                if response and hasattr(response, 'choices'):
                    # print(response.choices[0].message.content)
                    return (
                        response.choices[0].logprobs.content,
                        None,
                        response,
                    )
                print(
                    'Error calling OpenAI API with error message: '
                    + response["error"]["message"]
                )
                time.sleep(wait_seconds)
                # wait_seconds *= 2
            except Exception as e:  # pylint: disable=broad-exception-caught
                # Want to catch all exceptions happened during LLM calls.
                time.sleep(wait_seconds)
                # wait_seconds *= 2
                counter -= 1
                print('Error calling LLM, will retry soon...')
                print(e)
        return ERROR_CALLING_LLM, None, None

    def predict_logits_batch(self, text_prompts: list[str], images_list: list[list[np.ndarray]], max_token_len: int = 200
                             ) -> list[tuple[str, Optional[bool], Any]]:
        return [[]]

    def predict_scores_batch(self, text_prompts: list[str], images_list: list[list[np.ndarray]], max_token_len: int = 200
                             ) -> list[tuple[str, Optional[bool], Any]]:
        return [[]]


class Gpt4_Llama_Mix_Wrapper(LlmWrapper, MultimodalLlmWrapper):
    """OpenAI GPT4 wrapper.

    Attributes:
      openai_api_key: The class gets the OpenAI api key either explicitly, or
        through env variable in which case just leave this empty.
      max_retry: Max number of retries when some error happens.
      temperature: The temperature parameter in LLM to control result stability.
      model: GPT model to use based on if it is multimodal.
    """

    RETRY_WAITING_SECONDS = 1

    def __init__(
        self,
        model_name: str,
        service_name: str,
        local_model_name: str,
        max_retry: int = 100,
        temperature: float = 0.0,
        reward_type: str = "score",
        adapter_dir: str = "V-Droid-110K",
        prefix_sharing: bool = True,
    ):
        if max_retry <= 0:
            max_retry = 3
            print('Max_retry must be positive. Reset it to 3')
        self.max_retry = max_retry
        self.temperature = temperature
        self.model_name = model_name
        self.local_model_name = local_model_name
        self.reward_type = reward_type
        self.service_name = service_name
        adapter_dir = "MaginaDai/" + adapter_dir
        self.reward_model = LlamaRewardModel(
            local_model_name, adapter_dir, cluster=0, reward_type=reward_type, prefix_sharing=prefix_sharing)

    @classmethod
    def encode_image(cls, image: np.ndarray) -> str:
        return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

    def reset_kv_cache(self,):
        self.reward_model.reset_kv_cache()
        return

    def predict(
        self,
        text_prompts: str,
        images_list: list = None,
    ) -> tuple[str, Optional[bool], Any]:
        results = []
        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS

        while counter > 0:
            # try:
            batch_prompts = ["<|begin_of_text|>" +
                             text_prompt for text_prompt in text_prompts]

            # for text_prompt, images in zip(text_prompts, images_list):
            #     prompt = "<|begin_of_text|>" + text_prompt
            #     for _ in range(len(images)):
            #         prompt = "<|image|>" + prompt
            #     batch_prompts.append(prompt)

            with torch.no_grad():
                response = self.reward_model.generate_response_based_on_prompt(
                    batch_prompts)
            return response, None, None

        return [(ERROR_CALLING_LLM, None, None)] * len(text_prompts)

    def predict_mm(
        self, text_prompt: str, images: list[np.ndarray]
    ) -> tuple[str, Optional[bool], Any]:

        payload = {
            "model": self.model_name,
            "temperature": self.temperature,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": text_prompt},
                ],
            }],
            "max_tokens": 1000,
        }

        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS

        if self.service_name == 'openai':
            client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_ENDPOINT)
            self.deployment = OPENAI_MODEL_NAME

        elif self.service_name == 'trapi':
            scope = "api://trapi/.default"
            credential = get_bearer_token_provider(AzureCliCredential(), scope)
            client = AzureOpenAI(
                azure_endpoint=TRAPI_ENDPOINT,
                azure_ad_token_provider=credential,
                api_version=TRAPI_API_VERSION,
            )
            self.deployment = TRAPI_MODEL_NAME

        elif self.service_name == 'azure_openai':
            client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_OPENAI_ENDPOINT,
                azure_deployment=AZURE_OPENAI_MODEL_NAME
            )
            self.deployment = AZURE_OPENAI_MODEL_NAME

        else:
            raise ValueError('Unknown service name: %s' % self.service_name)

        while counter > 0:
            try:
                response = client.chat.completions.create(
                    messages=payload["messages"],
                    model=self.deployment,
                    temperature=self.temperature,
                    max_tokens=1000,
                )
                if response and hasattr(response, 'choices'):
                    # print(response.choices[0].message.content)
                    return (
                        response.choices[0].message.content,
                        None,
                        response,
                    )
                print(
                    'Error calling OpenAI API with error message: '
                    + response["error"]["message"]
                )
                time.sleep(wait_seconds)
            except Exception as e:  # pylint: disable=broad-exception-caught
                time.sleep(wait_seconds)
                counter -= 1
                print('Error calling LLM, will retry soon...')
                print(e)
        return ERROR_CALLING_LLM, None, None

    def predict_logits(self, text_prompt: str, images: list[np.ndarray], max_token_len: int = 200,
                       ) -> tuple[str, Optional[bool], Any]:
        return None

    def predict_logits_batch(self, text_prompts: list[str], images_list: list[list[np.ndarray]], max_token_len: int = 200
                             ) -> list[tuple[str, Optional[bool], Any]]:

        return None

    def predict_scores_batch(self, text_prompts: list[str], images_list: list[list[np.ndarray]] = None, max_token_len: int = 200
                             ) -> list[tuple[str, Optional[bool], Any]]:
        """
        Batched processing for score assignement.

        Args:
            text_prompts: List of text prompts.
            images_list: List of lists of images corresponding to each text prompt.
            max_token_len: Maximum token length for generated output.

        Returns:
            List of tuples (response, None, output) for each prompt in the batch.
        """

        results = []
        counter = self.max_retry
        wait_seconds = self.RETRY_WAITING_SECONDS
        while counter > 0:
            batch_prompts = ["<|begin_of_text|>" +
                             text_prompt for text_prompt in text_prompts]

            with torch.no_grad():
                if self.reward_type == "logits":
                    yes_no_logits = self.reward_model.get_next_token_logits_from_prompt(
                        batch_prompts)
                elif self.reward_type == "probs":
                    yes_no_logits = self.reward_model.get_next_token_probabilities_from_prompt(
                        batch_prompts)
                elif self.reward_type == "score":
                    yes_no_logits = self.reward_model.get_next_token_score_from_prompt(
                        batch_prompts)
                elif self.reward_type == "cot_score":
                    yes_no_logits = self.reward_model.get_next_token_score_cot_from_prompt(
                        batch_prompts)
                yes_no_logits = yes_no_logits.cpu().numpy()

            results = [(yes_no_logits[i, 0], None, yes_no_logits[i, :])
                       for i in range(len(batch_prompts))]
            return results

        return [(ERROR_CALLING_LLM, None, None)] * len(text_prompts)
