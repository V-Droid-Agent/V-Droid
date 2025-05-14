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
from ollama import Client
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
    # 'gpt-35-turbo',
    # 'gpt-35-turbo-16k',
    # 'gpt-35-turbo-instruct',
    # 'gpt-4',
    # 'gpt-4-32k_0613',
    'gpt-4_0125-Preview',
    # 'gpt-4o_2024-08-06',
    'gpt-4o_2024-11-20',
    # 'gpt-4o_2024-05-13',
    'gpt-4o-mini_2024-07-18',
    'deepseek-r1',
    # 'gpt-4-turbo',
    # 'gpt-4-turbo-v'
]

# deployment = trapi_deployment_list[0]
TRAPI_END_POINT = 'gcr/shared'
TRAPI_GCR_END_POINT = 'gcr/shared'
TRAPI_GCR_BASE_URL = f'https://trapi.research.microsoft.com/' + TRAPI_GCR_END_POINT
TRAPI_API_VERSION = '2024-10-21'
TRAPI_BASE_URL_2 = "https://gcraoai5sw2.openai.azure.com/"

TRAPI_API_KEY = ""
TRAPI_API_KEY_2 = ""
TRAPI_API_KEY_3 = ""
 

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
    self.generation_config=generation_types.GenerationConfig(
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
            [text_prompt] + [Image.fromarray(image) for image in images],
            safety_settings=None
            if enable_safety_checks
            else SAFETY_SETTINGS_BLOCK_NONE,
            generation_config=self.generation_config,
        )
        return output.text, True, output
      except Exception as e: 
        counter -= 1
        print(f'Error calling LLM, will retry in {retry_delay} seconds')
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
        print(f'Error calling LLM, will retry in {retry_delay} seconds')
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


class OllamaWrapper(LlmWrapper, MultimodalLlmWrapper):
  """Ollama wrapper.

  Attributes:
    max_retry: Max number of retries when some error happens.
    temperature: The temperature parameter in LLM to control result stability.
    model: GPT model to use based on if it is multimodal.
  """
  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str,
      max_retry: int = 3,
      temperature: float = 0.0,
  ):
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    self.max_retry = min(max_retry, 5)
    self.temperature = temperature
    self.model = model_name
    self.client = Client(host='http://localhost:11434')

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [])

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray]
  ) -> tuple[str, Optional[bool], Any]:

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS

    while counter > 0:
      try:
        response = self.client.chat(model=self.model, messages=[
        {
            'role': 'user',
            'content': text_prompt,
            'temperature': self.temperature,
        },
        ])
        if response and hasattr(response, 'choices'):
          # print(response.choices[0].message.content)
          return (
              response['message']['content'],
              None,
              response,
          )
        print(
            'Error calling OpenAI API with error message: '
            + response["error"]["message"]
        )
        time.sleep(wait_seconds)
        wait_seconds *= 2
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Want to catch all exceptions happened during LLM calls.
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
        print('Error calling LLM, will retry soon...')
        print(e)
    return ERROR_CALLING_LLM, None, None

def store_screen(state, step_idx, save_dir):
  # pdb.set_trace()
  save_path = os.path.join(save_dir, f"screen_shot/")
  if not os.path.exists(save_path):
    os.makedirs(save_path)

  pixels = state['raw_screenshot']
  img = Image.fromarray(np.uint8(pixels))
  img.save(save_path + f"step{step_idx}.jpg", 'JPEG')
  
  pixels_ann = state['before_screenshot_with_som']
  img_ann = Image.fromarray(np.uint8(pixels_ann))
  img_ann.save(save_path + f"step{step_idx}_ann.jpg", 'JPEG')
  return


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
      max_retry: int = 100,
      temperature: float = 0.0,
  ):
    # if 'OPENAI_API_KEY' not in os.environ:
    #   raise RuntimeError('OpenAI API key not set.')
    # self.openai_api_key = os.environ['OPENAI_API_KEY']
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')
    # self.max_retry = min(max_retry, 5)
    self.max_retry = max_retry
    self.temperature = temperature
    self.model_name = model_name

    if self.model_name == "gpt-4o":
      self.deployment = trapi_deployment_list[1]
    elif self.model_name == "gpt-4o-mini":
      self.deployment = trapi_deployment_list[2]
    elif self.model_name == "deepseek-r1":
      self.deployment = trapi_deployment_list[-1]
    else:
      self.deployment = trapi_deployment_list[0]

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
      measure_time = False,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, [], measure_time)

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray], measure_time = False
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

    if "deepseek" in self.model_name:
      client = OpenAI(api_key="sk-ce2f8583704e4741989757b210dd9b50",
                        base_url="https://api.deepseek.com")
        
    else:
      scope = "api://trapi/.default"
      credential = get_bearer_token_provider(AzureCliCredential(),scope)
      client = AzureOpenAI(
          azure_endpoint=TRAPI_GCR_BASE_URL,
          azure_ad_token_provider=credential,
          api_version=TRAPI_API_VERSION,
      )

    # client = AzureOpenAI(
    #     api_key=TRAPI_API_KEY_2,
    #     api_version=TRAPI_API_VERSION,
    #     azure_endpoint=TRAPI_GCR_BASE_URL,
    #     azure_deployment=self.deployment
    # )

    # client = AzureOpenAI(
    #     api_key=TRAPI_API_KEY_2,
    #     api_version=TRAPI_API_VERSION,
    #     azure_endpoint=TRAPI_BASE_URL_2,
    #     azure_deployment=self.deployment
    # )
    # pdb.set_trace()
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
    scope = "api://trapi/.default"
    credential = get_bearer_token_provider(AzureCliCredential(),scope)
    client = AzureOpenAI(
        azure_endpoint=TRAPI_GCR_BASE_URL,
        azure_ad_token_provider=credential,
        api_version=TRAPI_API_VERSION,
    )
    
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
  

class LlamaVisionWrapper(LlmWrapper, MultimodalLlmWrapper):
  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str,
      max_retry: int = 1,
      temperature: float = 0.2,
  ):
    self.model_name = model_name
    # quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    if "Llama-3.2" in model_name:
        self.model = MllamaForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
        )
    elif "Llama-3.1" in model_name:
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
        )

    # self.model = MllamaForConditionalGeneration.from_pretrained(
    #   model_name,
    #   device_map="auto",
    # )

    self.processor = AutoProcessor.from_pretrained(model_name)
    self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    self.max_retry = max_retry
    self.temperature = temperature # banned here

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompt: str,
      images: list[np.ndarray],
      max_token_len: int=200,
  ) -> tuple[str, Optional[bool], Any]:
    return self.predict_mm(text_prompt, images, max_token_len)

  def predict_mm(
      self, text_prompt: str, images: list[np.ndarray], max_token_len: int=200,
  ) -> tuple[str, Optional[bool], Any]:

    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS
    
    prompt = "<|begin_of_text|>" + text_prompt
    for _ in range(len(images)):
      prompt = "<|image|>" + prompt

    generation_args = {
        "max_new_tokens": max_token_len,
        "temperature": self.temperature,
        "do_sample": True,
    }

    while counter > 0:
      try:
        # pdb.set_trace()
        with torch.no_grad():
          if images:
            inputs = self.processor(images, prompt, return_tensors="pt").to('cuda')
          else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
          output = self.model.generate(**inputs, 
                                      **generation_args)
          response = self.processor.decode(output[0][len(inputs["input_ids"][0]):], skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # response[len(inputs["input_ids"][0])]
        # output[0][len(inputs["input_ids"][0]):]
        if response:
          return (
              response,
              None,
              output,
          )
        time.sleep(wait_seconds)
        wait_seconds *= 2
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Want to catch all exceptions happened during LLM calls.
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
        print('Error calling LLM, will retry soon...')
        print(e)
    return ERROR_CALLING_LLM, None, None

  def predict_logits(self, text_prompt: str, images: list[np.ndarray], max_token_len: int=200,
  ) -> tuple[str, Optional[bool], Any]:
  
    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS

    prompt = "<|begin_of_text|>" + text_prompt
    for _ in range(len(images)):
      prompt = "<|image|>" + prompt

    while counter > 0:
      try:
        # pdb.set_trace()
        with torch.no_grad():
          if images:
            inputs = self.processor(images, prompt, return_tensors="pt").to('cuda')
          else:
            inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
            # inputs = self.processor(text_prompt, return_tensors="pt").to(self.model.device)
          output = self.model.generate(**inputs, 
                                      # do_sample=False, 
                                      max_new_tokens=max_token_len, 
                                      output_scores=True, 
                                      return_dict_in_generate=True)
          response = [self.tokenizer.decode(seq[len(inputs["input_ids"][0]):], skip_special_tokens=True) for seq in output['sequences']]
          scores = torch.stack(output["scores"], dim=0)
          softmax = torch.nn.LogSoftmax(dim=-1)
          probs = softmax(scores)

          # max_values, max_indices = torch.max(probs, dim=-1)
          # output["logprob"] = max_values.reshape(-1).cpu().numpy()

          probs = probs.squeeze(1)
          max_indices = output['sequences'][0][len(inputs["input_ids"][0]):].squeeze(-1)
          token_positions = torch.arange(len(max_indices))
          output["logprob"] = probs[token_positions, max_indices].cpu().numpy()
         
          decoded_tokens = [self.tokenizer.decode([idx.item()]) for idx in max_indices]
          output["decoded_tokens"] = decoded_tokens
          
        
        # pdb.set_trace()
        if output:
          return (
              response[0],
              None,
              output,
          )
        time.sleep(wait_seconds)
        wait_seconds *= 2
      except Exception as e:  # pylint: disable=broad-exception-caught
        # Want to catch all exceptions happened during LLM calls.
        time.sleep(wait_seconds)
        wait_seconds *= 2
        counter -= 1
        print('Error calling LLM, will retry soon...')
        print(e)
    return ERROR_CALLING_LLM, None, None
  

  def predict_mm_batch(
        self, text_prompts: list[str], images_list: list[list[np.ndarray]], max_token_len: int = 200,
    ) -> list[tuple[str, Optional[bool], Any]]:
      """
      Process a batch of text prompts with optional image inputs.
      
      Args:
          text_prompts: List of text prompts to process.
          images_list: List of lists of images. Each inner list corresponds to a prompt.
          max_token_len: Maximum token length for generated output.

      Returns:
          A list of tuples, where each tuple contains the response, a flag, and the raw output for each prompt.
      """
      assert len(text_prompts) == len(images_list), "Number of text prompts and image lists must be equal."
      
      results = []
      counter = self.max_retry
      wait_seconds = self.RETRY_WAITING_SECONDS

      generation_args = {
          "max_new_tokens": max_token_len,
          "temperature": self.temperature,
          "do_sample": True,
      }

      while counter > 0:
          try:
              batch_prompts = []
              for text_prompt, images in zip(text_prompts, images_list):
                  prompt = "<|begin_of_text|>" + text_prompt
                  for _ in range(len(images)):
                      prompt = "<|image|>" + prompt
                  batch_prompts.append(prompt)
              
              with torch.no_grad():
                  if any(images_list):  # If there are images in any batch entry
                      batch_inputs = self.processor(images_list, batch_prompts, return_tensors="pt", padding=True).to('cuda')
                  else:  # If no images in any batch entry
                      batch_inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to('cuda')

                  batch_outputs = self.model.generate(**batch_inputs, **generation_args)

                  for i, output in enumerate(batch_outputs):
                      response = self.processor.decode(output[len(batch_inputs["input_ids"][i]):], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                      results.append((response, None, output))
              return results
          
          except Exception as e:  # Catch all exceptions during LLM calls
              print('Error calling LLM for batch, will retry soon...')
              print(e)
              time.sleep(wait_seconds)
              wait_seconds *= 2
              counter -= 1

      # If retries are exhausted, return errors for the entire batch
      return [(ERROR_CALLING_LLM, None, None)] * len(text_prompts)

  def predict_logits_batch(self, text_prompts: list[str], images_list: list[list[np.ndarray]], max_token_len: int = 200
    ) -> list[tuple[str, Optional[bool], Any]]:
      """
      Batched processing for logits predictions.

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
          # try:
          batch_prompts = []
          for text_prompt, images in zip(text_prompts, images_list):
              prompt = "<|begin_of_text|>" + text_prompt
              for _ in range(len(images)):
                  prompt = "<|image|>" + prompt
              batch_prompts.append(prompt)

          with torch.no_grad():
              if any(images_list):  # If any entry has images
                  batch_inputs = self.processor(images_list, batch_prompts, return_tensors="pt", padding=True).to('cuda')
              else:  # If no images in any batch entry
                  batch_inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to('cuda')

              batch_outputs = self.model.generate(
                  **batch_inputs,
                  max_new_tokens=max_token_len,
                  output_scores=True,
                  return_dict_in_generate=True,
              )

              batch_responses = [
                  self.tokenizer.decode(seq[len(batch_inputs["input_ids"][i]):], skip_special_tokens=True)
                  for i, seq in enumerate(batch_outputs['sequences'])
              ]

              batch_scores = torch.stack(batch_outputs["scores"], dim=0)
              softmax = torch.nn.LogSoftmax(dim=-1)
              probs = softmax(batch_scores)

              # max_values, max_indices = torch.max(probs, dim=-1)
              # logprobs = max_values.T.cpu().numpy()

              # decoded_tokens = [
              #     [self.tokenizer.decode([idx.item()], skip_special_tokens=True) for idx in max_indices[:, i]]
              #     for i in range(max_indices.shape[1])
              # ]
              
              # # Append results for each prompt in the batch
              # for i in range(len(text_prompts)):
              #     output = {
              #         "logprob": logprobs[i],
              #         "decoded_tokens": decoded_tokens[i],
              #     }
              #     results.append((batch_responses[i], None, output))
              
              for i in range(len(text_prompts)):
                  output = {
                      "logprob": None,
                      "decoded_tokens": None,
                  }

                  max_indices = batch_outputs['sequences'][i][len(batch_inputs["input_ids"][i]):].squeeze(-1)
                  token_positions = torch.arange(len(max_indices))
                  output["logprob"] = probs[token_positions, i, max_indices].cpu().numpy()
                  output["decoded_tokens"] = [self.tokenizer.decode([idx.item()]) for idx in max_indices]
                  results.append((batch_responses[i], None, output))
                  # pdb.set_trace()
          return results

          # except Exception as e:  # Catch all exceptions during LLM calls
          #     print('Error calling LLM for batch, will retry soon...')
          #     print(e)
          #     time.sleep(wait_seconds)
          #     wait_seconds *= 2
          #     counter -= 1

      # If retries are exhausted, return errors for the entire batch
      return [(ERROR_CALLING_LLM, None, None)] * len(text_prompts)


# @torch.no_grad()
# def query_next_token(llamamodel, tokenizer, prompts, yes_no):
#   if isinstance(prompts, str):
#       prompts = [prompts]
#   ret = []
#   for prompt in prompts:
#       tokens = tokenizer.encode(prompt, bos=True, eos=False)
#       tokens = torch.tensor([tokens]).cuda().long()
#       output, h = llamamodel.model.forward(tokens, start_pos=0)
#       ret.append(output)
#   outputs = torch.cat(ret, dim=0)
#   filtered = outputs[:, yes_no]
#   dist = torch.softmax(filtered, dim=-1)
#   return dist


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
      local_model_name: str,
      max_retry: int = 100,
      temperature: float = 0.0,
      reward_type: str = "score",
      adapter_dir: str = "RewardModel_Ori", ##  "LlamaReward", "RewardModel_Ori"
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

    if self.model_name == "gpt-4o":
      self.deployment = trapi_deployment_list[1]
    elif self.model_name == "gpt-4o-mini":
      self.deployment = trapi_deployment_list[2]
    else:
      self.deployment = trapi_deployment_list[0]

    adapter_dir = "MaginaDai/" + adapter_dir
    self.reward_model = LlamaRewardModel(local_model_name, adapter_dir, cluster=0, reward_type=reward_type, prefix_sharing=prefix_sharing)

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
        batch_prompts = ["<|begin_of_text|>" + text_prompt for text_prompt in text_prompts]

        # for text_prompt, images in zip(text_prompts, images_list):
        #     prompt = "<|begin_of_text|>" + text_prompt
        #     for _ in range(len(images)):
        #         prompt = "<|image|>" + prompt
        #     batch_prompts.append(prompt)

        with torch.no_grad():
            response = self.reward_model.generate_response_based_on_prompt(batch_prompts)
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

    scope = "api://trapi/.default"
    credential = get_bearer_token_provider(AzureCliCredential(),scope)
    client = AzureOpenAI(
        azure_endpoint=TRAPI_GCR_BASE_URL,
        azure_ad_token_provider=credential,
        api_version=TRAPI_API_VERSION,
    )
    
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
  
  def predict_logits(self, text_prompt: str, images: list[np.ndarray], max_token_len: int=200,
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
          batch_prompts = ["<|begin_of_text|>" + text_prompt for text_prompt in text_prompts]

          with torch.no_grad():
              if self.reward_type == "logits":
                yes_no_logits = self.reward_model.get_next_token_logits_from_prompt(batch_prompts)
              elif self.reward_type == "probs":
                yes_no_logits = self.reward_model.get_next_token_probabilities_from_prompt(batch_prompts)
              elif self.reward_type == "score":
                yes_no_logits = self.reward_model.get_next_token_score_from_prompt(batch_prompts)
              elif self.reward_type == "cot_score":
                yes_no_logits = self.reward_model.get_next_token_score_cot_from_prompt(batch_prompts)
              yes_no_logits = yes_no_logits.cpu().numpy()
          
          results = [(yes_no_logits[i, 0], None, yes_no_logits[i, :]) for i in range(len(batch_prompts))]
          return results
      
      return [(ERROR_CALLING_LLM, None, None)] * len(text_prompts)
  

class Gpt4_Policy_Lora_Mix_Wrapper(LlmWrapper, MultimodalLlmWrapper):
  RETRY_WAITING_SECONDS = 20

  def __init__(
      self,
      model_name: str,
      local_model_name: str,
      max_retry: int = 100,
      temperature: float = 0.0,
      adapter_dir: str = "RewardModel_Ori",
      agent_name: str = "T3A",
  ):
    if max_retry <= 0:
      max_retry = 3
      print('Max_retry must be positive. Reset it to 3')

    self.max_retry = max_retry
    self.temperature = temperature
    self.model_name = model_name
    self.local_model_name = local_model_name

    if self.model_name == "gpt-4o":
      self.deployment = trapi_deployment_list[1]
    elif self.model_name == "gpt-4o-mini":
      self.deployment = trapi_deployment_list[2]
    else:
      self.deployment = trapi_deployment_list[0]

    adapter_dir = "MaginaDai/" + adapter_dir
    self.policy_model = load_lora_model_from_dir(local_model_name, adapter_dir, kv_cache=True, enable_prefix_caching=False)
    self.lora_request=LoRARequest("adapter", 1, adapter_dir)

    self.analysis_params = SamplingParams(temperature=self.temperature, 
                                            max_tokens=1 if "selector" in agent_name else 200,
                                            logprobs=1,
                                            truncate_prompt_tokens=2800)

  @classmethod
  def encode_image(cls, image: np.ndarray) -> str:
    return base64.b64encode(array_to_jpeg_bytes(image)).decode('utf-8')

  def predict(
      self,
      text_prompts: str,
  ) -> tuple[str, Optional[bool], Any]:
    results = []
    counter = self.max_retry
    wait_seconds = self.RETRY_WAITING_SECONDS

    while counter > 0:
        prompt = "<|begin_of_text|>" + text_prompts

        with torch.no_grad():
            outputs = self.policy_model.generate(prompt, sampling_params=self.analysis_params,lora_request=self.lora_request)
        return outputs[0].outputs[0].text, None, None
    
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

    scope = "api://trapi/.default"
    credential = get_bearer_token_provider(AzureCliCredential(),scope)
    client = AzureOpenAI(
        azure_endpoint=TRAPI_GCR_BASE_URL,
        azure_ad_token_provider=credential,
        api_version=TRAPI_API_VERSION,
    )
    
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