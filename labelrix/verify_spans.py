import os
import json
import re
import numpy as np
from typing import List, Dict, Any, Tuple, TypedDict, Literal
from collections import defaultdict
from const import VERIFICATION_PROMPT_TEMPLATES, PII_EXTRA_RULES_VERIFICATION
from langfuse.openai import OpenAI
from dotenv import load_dotenv
import time
from pathlib import Path
from segment_lines import load_textract_pages, extract_line_segments
from mistralai import Mistral
from langfuse import observe, get_client
from concurrent.futures import ThreadPoolExecutor, as_completed
langfuse = get_client()


load_dotenv(dotenv_path="/Volumes/MyDataDrive/thesis/code-2/.env")

# Initialize Nebius client
nebius_client = OpenAI(
    api_key=os.getenv("NEBIUS_API_KEY"),
    base_url="https://api.studio.nebius.com/v1/",
)

# Initialize OpenAI client
deepinfra_client = OpenAI(
    api_key=os.getenv("DEEPINFRA_API_KEY"),
    base_url="https://api.deepinfra.com/v1/openai"
)

mistral_client = Mistral(api_key=os.environ["MISTRAL_API_KEY"])


# Model configurations
MODEL_CONFIGS = {
    "Qwen3-30B-A3B": {
        "client": "nebius",
        "model": "Qwen/Qwen3-30B-A3B",
        "temperature": 0.6,
        "max_tokens": 32000
    },
    "deepseek-ai/DeepSeek-R1-0528": {
        "client": "nebius",
        "model": "deepseek-ai/DeepSeek-R1-0528",
        "temperature": 0.6,
        "max_tokens": 32000
    },  
    "magistral-small-latest": {
        "client": "mistral",
        "model": "magistral-small-latest",  # or whatever the actual model name is
        "temperature": 0.6,
        "max_tokens": 32000
    }
}


@observe(as_type="generation")
def mistral_completion(**kwargs):
  # Clone kwargs to avoid modifying the original input
  kwargs_clone = kwargs.copy()
 
  # Extract relevant parameters from kwargs
  input = kwargs_clone.pop('messages', None)
  model = kwargs_clone.pop('model', None)
  min_tokens = kwargs_clone.pop('min_tokens', None)
  max_tokens = kwargs_clone.pop('max_tokens', None)
  temperature = kwargs_clone.pop('temperature', None)
  top_p = kwargs_clone.pop('top_p', None)
 
  # Filter and prepare model parameters for logging
  model_parameters = {
        "maxTokens": max_tokens,
        "minTokens": min_tokens,
        "temperature": temperature,
        "top_p": top_p
    }
  model_parameters = {k: v for k, v in model_parameters.items() if v is not None}
 
  # Log the input and model parameters before calling the LLM
  langfuse.update_current_generation(
      input=input,
      model=model,
      model_parameters=model_parameters,
      metadata=kwargs_clone,
 
  )
 
  # Call the Mistral model to generate a response
  res = mistral_client.chat.complete(**kwargs)
 
  # Log the usage details and output content after the LLM call
  langfuse.update_current_generation(
      usage_details={
          "input": res.usage.prompt_tokens,
          "output": res.usage.completion_tokens
      },
      output=res.choices[0].message.content
  )
 
  # Return the model's response object
  return res

def call_model_api(client: str, model: str, messages: Any, temperature: float, max_tokens: int) -> str:
    """
    Make the actual API call to the model with retries
    
    Args:
        client: Which client to use ('nebius' or 'openai')
        model: Name of the model to use
        messages: List of message dictionaries
        temperature: Temperature parameter for generation
        max_tokens: Maximum tokens to generate
        
    Returns:
        Model's response text
        
    Raises:
        ValueError: If response is empty or client type is unknown
    """
    max_attempts = 3
    base_wait = 4  # seconds
    
    for attempt in range(max_attempts):
        try:
            if client == "nebius":
                response = nebius_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=False,
                    top_p=0.95
                )
            elif client == "deepinfra":
                response = deepinfra_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    stream=False,
                )
            elif client == "mistral":
                response = mistral_completion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                )
            else:
                raise ValueError(f"Unknown client type: {client}")
                
            if not response.choices or not response.choices[0].message.content:
                raise ValueError("Empty response from model")
                
            return response.choices[0].message.content
            
        except Exception as e:
            if attempt == max_attempts - 1:  # Last attempt
                print(f"Failed after {max_attempts} attempts. Last error: {e}")
                raise
            
            # Calculate wait time: 4s, 8s, 16s but cap at 10s
            wait_time = 0.1
            print(f"Attempt {attempt + 1} failed, waiting {wait_time}s before retry. Error: {e}")
            time.sleep(wait_time)
    
    # This should never happen since we either return or raise in the loop
    raise RuntimeError("Unexpected error: reached end of retry loop")

def get_model_response(model: str, prompt: str) -> str:
    """
    Get verification response from specified model
    
    Args:
        model: Name of the model to use
        prompt: Formatted prompt to send to model
        
    Returns:
        Model's response text
    """
    config = MODEL_CONFIGS[model]
    messages = [
        {"role": "system", "content": "You are a helpful assistant that verifies PII entities in documents."},
        {"role": "user", "content": prompt}
    ]
    
    return call_model_api(
        client=config["client"],
        model=config["model"],
        messages=messages,
        temperature=config["temperature"],
        max_tokens=config["max_tokens"]
    )

def parse_batch_verification_response(response: str, batch_size: int) -> Tuple[List[int], bool]:
    """
    Parse model response for batch verification into individual votes
    
    Args:
        response: Raw response text from model in the format:
                 <json-start>
                 {"Verifications": ["1:Yes", "2:No", "3:Yes"]}
                 <json-end>
        batch_size: Number of spans in the batch
        
    Returns:
        Tuple of (votes list, success boolean)
        - votes: List of verification votes (0 or 1) for each span
        - success: Whether parsing was successful
    """
    try:
        # Strip <think>…</think> blocks that some models might prepend
        cleaned_resp = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

        # Extract JSON between tags
        json_start = cleaned_resp.find("<json-start>") + len("<json-start>")
        json_end = cleaned_resp.find("<json-end>")
        
        if json_start == -1 or json_end == -1:
            print("JSON tags not found in response")
            return [0] * batch_size, False
            
        json_str = cleaned_resp[json_start:json_end].strip()
        
        # Parse JSON
        result = json.loads(json_str)
        verifications = result["Verifications"]
        
        # Convert to binary votes
        votes = [0] * batch_size
        for v in verifications:
            idx, decision = v.split(":")
            idx = int(idx) - 1  # Convert to 0-based index
            if 0 <= idx < batch_size:  # Validate index
                votes[idx] = 1 if decision.strip().lower() == "yes" else 0
            else:
                print(f"Warning: Invalid index {idx+1} in model response")
                return votes, False
            
        return votes, True
        
    except Exception as e:
        print(f"Error parsing response: {e}")
        return [0] * batch_size, False  # Return all negatives and False for parse failure

def get_and_parse_model_response(model_name: str, batch_prompt: str, batch_size: int) -> List[int]:
    """
    Call the model and attempt to parse its JSON response. We will make **up to
    `max_attempts` separate API calls**, but **only** when the previous call
    *succeeded* yet produced un-parsable JSON.  If the API itself errors out
    (network, rate-limit, etc.) we do **not** issue another call – we simply
    return negative votes because we have no response to work with.
    """

    max_attempts = 3
    base_wait = 4  # seconds for back-off between parse retries

    for attempt in range(max_attempts):
        try:
            # One API call (call_model_api already has its own transient-error retries)
            response = get_model_response(model_name, batch_prompt)
        except Exception as api_err:
            # API failed – do NOT retry with another call as per requirement.
            print(f"[{model_name}] API error: {api_err}. No further retries.")
            return [0] * batch_size

        # Got a response – attempt to parse it
        votes, success = parse_batch_verification_response(response, batch_size)
        if success:
            return votes  # All good!

        # Parsing failed – decide whether to try again
        if attempt < max_attempts - 1:
            wait_time = 0.1
            print(f"[{model_name}] Parsing failed (attempt {attempt + 1}/{max_attempts})")
            time.sleep(wait_time)
        else:
            print(f"[{model_name}] Parsing failed after {max_attempts} attempts. Giving up.")

    # All parse retries exhausted – return negative votes
    return [0] * batch_size

def verify_spans_in_file(file_spans: List[Dict[str, Any]], spans_by_type: Dict, document_text: str) -> List[Dict[str, Any]]:
    """
    Verify spans within a single file, batching by PII type.

    For each span we build a **flat vote array** of length
        `len(MODEL_CONFIGS) * len(VERIFICATION_PROMPT_TEMPLATES)`.
    Indexing layout: model-major order →
        index = model_idx * num_templates + template_idx
    where `model_idx` follows the order returned by `MODEL_CONFIGS.items()` and
    `template_idx` is the index into `VERIFICATION_PROMPT_TEMPLATES`.
    """

    num_models = len(MODEL_CONFIGS)
    num_templates = len(VERIFICATION_PROMPT_TEMPLATES)
    total_slots = num_models * num_templates

    # Ensure vote array exists on each span
    for span in file_spans:
        span["votes"] = [0] * total_slots  # initialise with zeros

    for pii_type, type_spans in spans_by_type.items():
        batch_texts = [rec["value"] for _, rec in type_spans]
        batch_indices = [idx for idx, _ in type_spans]

        # Launch all (model, template) calls in parallel
        futures = {}
        with ThreadPoolExecutor(max_workers=16) as pool:  # adjust workers as needed
            for model_idx, (model_name, _) in enumerate(MODEL_CONFIGS.items()):
                for template_idx, template in enumerate(VERIFICATION_PROMPT_TEMPLATES):
                    prompt = format_batch_prompt(template, pii_type, batch_texts, document_text)
                    fut = pool.submit(
                        get_and_parse_model_response,
                        model_name,
                        prompt,
                        len(batch_texts),
                    )
                    futures[fut] = (model_idx, template_idx)

            # Collect results as they complete
            for fut in as_completed(futures):
                model_idx, template_idx = futures[fut]
                try:
                    votes = fut.result()
                except Exception as exc:
                    print(f"Error in parallel fetch: {exc}")
                    votes = [0] * len(batch_texts)

                flat_index = model_idx * num_templates + template_idx
                for local_i, global_i in enumerate(batch_indices):
                    file_spans[global_i]["votes"][flat_index] = votes[local_i]

    # # Derive per-span verifiers (models with at least one positive template vote)
    # verified = []
    # min_votes = 1  # keep span if at least one positive vote

    # for span in file_spans:
    #     votes_arr = span["votes"]
    #     positive = sum(votes_arr)

    #     # Determine verifiers: a model is positive if any of its template votes are 1
    #     verifiers = []
    #     for m_idx, model_name in enumerate(MODEL_CONFIGS.keys()):
    #         model_slice = votes_arr[m_idx*num_templates : (m_idx+1)*num_templates]
    #         if any(model_slice):
    #             verifiers.append(model_name)
    #     span["verifiers"] = verifiers

    #     if positive >= min_votes:
    #         verified.append(span)

    # Return all spans; caller can decide what to do with the votes
    return file_spans

def format_batch_prompt(template: str, pii_type: str, spans: List[str], document_text: str) -> str:
    """
    Format a prompt for batch verification of spans using the template from const.py
    
    Args:
        template: Prompt template to use
        pii_type: Type of PII being verified
        spans: List of span texts to verify
        document_text: Full text of the document to use as context
        
    Returns:
        Formatted prompt string for batch verification
    """
    # Format the span list as numbered items
    span_list = "\n".join(f"{i+1}. {span}" for i, span in enumerate(spans))
    
    # Format the prompt using the template
    return template.format(
        pii_type=pii_type,
        document=document_text,
        span_list=span_list,
        extra_rule=PII_EXTRA_RULES_VERIFICATION[pii_type]
    )

def build_page_text(textract_json_path: str, page_num: int) -> str:
    """Return plain text of a single page inside a Textract JSON file."""
    if not Path(textract_json_path).exists():
        print(f"⚠️  Textract file not found: {textract_json_path}. Using empty text.")
        return ""

    for pid, blocks in load_textract_pages(textract_json_path):
        if pid.endswith(f"_page{page_num}"):
            segments, _ = extract_line_segments(blocks)
            return "\n".join(s["text"] for s in segments)

    print(f"⚠️  Page {page_num} not found in {textract_json_path}. Using empty text.")
    return ""

def shard_path(doc_id: str, root_dir: str) -> str:
    """Return path like root_dir/f/j/j/c/<doc_id>/<doc_id>.json (sharded)."""
    prefix_dirs = list(doc_id[:4])
    return os.path.join(root_dir, *prefix_dirs, doc_id, f"{doc_id}.json")

def load_spans(votes_dir: str, textract_dir: str, out_dir: str):
    """
    Load spans from votes_dir and save them to out_dir
    
    Args:
        votes_dir: Directory containing vote files
        textract_dir: Directory containing Textract JSON files
        out_dir: Directory to save output files
    """
    # Process files one at a time
    all_verified_spans = []
    
    for fname in sorted(os.listdir(votes_dir)):
        if not (fname.startswith('votes_') and fname.endswith('.json')):
            continue
            
        out_path = Path(out_dir) / f"{fname}_verified.json"
        if out_path.exists():
            print(f"✅ {out_path.name} already exists – skipping.")
            continue

        path = os.path.join(votes_dir, fname)
        print(f"\nProcessing file: {fname}")
        
        # Load spans (model votes) for this file
        with open(path, 'r') as f:
            file_spans = json.load(f)
            
        if not file_spans:
            print(f"No spans found in {fname}")
            continue
            
        # Rebuild document text for this file
        base_name = fname[len('votes_'):-len('.json')]  # e.g. fjjc0186_page1

        if "_page" in base_name:
            doc_id, page_str = base_name.split("_page", 1)
            try:
                page_num = int(page_str)
            except ValueError:
                page_num = 1
        else:
            doc_id, page_num = base_name, 1

        textract_path = shard_path(doc_id, textract_dir)
        document_text = build_page_text(textract_path, page_num)

        # Group spans by PII type within this file
        spans_by_type = defaultdict(list)
        for idx, span in enumerate(file_spans):
            key = span.get('pii_type')
            if isinstance(key, list):
                # If there's more than one item, pick the first; else None
                key = key[0] if key else None
            spans_by_type[key].append((idx, span))
            print(f"Found {key}: {span.get('value')}")
        
        # Run verification
        processed_spans = verify_spans_in_file(file_spans, spans_by_type, document_text)
        all_verified_spans.extend(processed_spans)

        out_path.parent.mkdir(exist_ok=True, parents=True)

        with out_path.open("w", encoding="utf-8") as f:
            json.dump(processed_spans, f, ensure_ascii=False, indent=2)

    