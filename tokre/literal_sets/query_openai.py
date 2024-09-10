import os

import backoff
from openai import OpenAI
import logging
import traceback


import tokre

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)


@backoff.on_exception(backoff.expo, Exception, max_time=120)
def query_openai(messages, model: str = "gpt-4o", temperature: float = 0.8, **kwargs):
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]

    if tokre._openai_api_key is not None:
        api_key = tokre._openai_api_key
    elif "OPENAI_API_KEY" in os.environ:
        api_key = os.environ["OPENAI_API_KEY"]
    else:
        raise ValueError(
            "Either the `OPENAI_API_KEY` environment variable needs to be set or `api_key` needs to be passed to `tokre.setup`."
        )
    client = OpenAI(api_key=api_key)
    try:
        response = client.chat.completions.create(
            model=model, messages=messages, temperature=temperature, **kwargs
        )
    except Exception as e:
        logger.error(f"Error in OpenAI API call: {str(e)}")
        logger.error(f"Full exception: {traceback.format_exc()}")
        raise
    return response.choices[0].message
