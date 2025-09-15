import os
import json
from typing import Tuple, List, Dict, Any
from typing import List, Tuple
from collections import defaultdict
import re
import textdistance
import pandas as pd
from ollama import chat
from ollama import ChatResponse
# from langfuse.openai import OpenAI
from together import Together
from openai import OpenAI

from dotenv import load_dotenv

import time

load_dotenv(dotenv_path="/Volumes/MyDataDrive/thesis/code-2/.env")

# Here we define local imports
import const

openai_api_key = ""
openai_api_base = "https://api.studio.nebius.com/v1/"

client = OpenAI(
    api_key=os.getenv("NEBIUS_API_KEY"),
    base_url="https://api.studio.nebius.com/v1/",
)

def normalize_text(s: str) -> str:
    s = s.lower().strip()
    # s = re.sub(r"[^a-z0-9]+", " ", s).strip()
    return s


def extract_line_segments(
    blocks: List[Dict[str, Any]], row_tol: float = 0.01, sep: str = "\t"
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    """
    Cluster LINE blocks into visual rows, then for each row:
      - sort left→right
      - merge their texts with `sep` (default “\t”)
      - union their bboxes
      - collect all child WORD IDs
    Returns (segments, block_map).
    """
    # 1) Build lookup
    block_map = {b["Id"]: b for b in blocks}

    # 2) Gather LINE blocks & their y-midpoints
    lines = []
    for b in blocks:
        if b.get("BlockType") != "LINE":
            continue
        bb = b["Geometry"]["BoundingBox"]
        ymid = bb["Top"] + bb["Height"] / 2
        lines.append({"block": b, "ymid": ymid})

    # 3) Cluster into rows by y-mid ± row_tol
    rows: List[List[Dict[str, Any]]] = []
    for item in sorted(lines, key=lambda x: x["ymid"]):
        placed = False
        for row in rows:
            if abs(row[0]["ymid"] - item["ymid"]) <= row_tol:
                row.append(item)
                placed = True
                break
        if not placed:
            rows.append([item])

    # 4) Build segments
    segments: List[Dict[str, Any]] = []
    for seg_id, row in enumerate(rows):
        # sort left→right
        row_blocks = [
            x["block"]
            for x in sorted(
                row, key=lambda x: x["block"]["Geometry"]["BoundingBox"]["Left"]
            )
        ]

        # merge text with tab separators
        text = sep.join(b["Text"] for b in row_blocks)

        # union bounding boxes
        bbs = [b["Geometry"]["BoundingBox"] for b in row_blocks]
        x0 = min(bb["Left"] for bb in bbs)
        y0 = min(bb["Top"] for bb in bbs)
        x1 = max(bb["Left"] + bb["Width"] for bb in bbs)
        y1 = max(bb["Top"] + bb["Height"] for bb in bbs)

        token = f"{int(x0*100)}|{int(y0*100)}"

        # collect WORD children
        word_ids: List[str] = []
        for b in row_blocks:
            for rel in b.get("Relationships", []):
                if rel.get("Type") != "CHILD":
                    continue
                for cid in rel.get("Ids", []):
                    child = block_map.get(cid)
                    if child and child.get("BlockType") == "WORD":
                        word_ids.append(cid)

        segments.append(
            {
                "id": seg_id,
                "text": text,
                "bbox": (x0, y0, x1, y1),
                "token": token,
                "word_ids": word_ids,
            }
        )

    return segments, block_map


def build_bbox_prompt(pii_type: str, segments: List[Dict[str, Any]], template) -> str:
    """
    segments[i]['text'] may contain '\t' between original columns.
    We list:
      id: text    (X|Y)
    and then ask for JSON of values with those tokens.
    """
    lines = ["Text", "Document lines (id: text (X|Y)):"]
    for s in segments:
        # s['text'] already has tabs between columns
        lines.append(f"{s['id']}: {s['text']} ({s['token']})")

    extra_rule = const.PII_EXTRA_RULES.get(pii_type, "")
    filled_prompt = template.format(pii_type=pii_type, extra_rule=extra_rule)
    lines.extend([filled_prompt])

    return "\n".join(lines)


def call_llm(
    model,
    prompt: str, max_new_tokens: int = 32000, temperature: float = 0.6, use_api=True
) -> str:
    messages = [{"role": "user", "content": prompt}]
    if use_api:
        # time.sleep(0.01)
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_completion_tokens=max_new_tokens,
            top_p=0.95,
            stream=False,
            stop=None,
        )
        content = response.choices[0].message.content
        return content
    else:
        # use olama
        response: ChatResponse = chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": temperature, "num_predict": max_new_tokens},
        )

        # Extract the assistant's reply
        raw = response.message.content
        return raw


def try_extract(response: str, segments, block_map) -> List[Dict[str, Any]]:
    """
    Attempts a single extract pass: find tags, parse JSON, normalize.
    Raises ValueError if tags missing, JSONDecodeError if invalid JSON.
    """
    cleaned = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    # 1) Find the JSON block
    match = re.search(r"<json-start>(.*?)<json-end>", cleaned, re.DOTALL)
    if not match:
        raise ValueError("No <json-start>…<json-end> block found")
    payload_str = match.group(1).strip()

    # 2) Parse JSON
    payload = json.loads(payload_str)  # may raise JSONDecodeError
    # print(payload)

    items = payload.get("values") if isinstance(payload, dict) else payload
    # print(items)
    if not isinstance(items, list):
        raise TypeError(f"Expected a list-of-dicts under 'values' or at top level, got {type(items)}")
    
    # 4) Hand off to your parser
    return parse_entity_values(segments, items, block_map)


def extract_with_retries(
    prompt, segments, model, block_map, max_tries: int = 3
) -> List[Dict[str, Any]]:
    """
    Builds a prompt, calls the LLM, and tries to extract the JSON.
    Retries on ValueError or JSONDecodeError up to max_tries times.
    """
    last_err = None
    for attempt in range(1, max_tries + 1):
        response = call_llm(prompt=prompt,model=model,temperature=0.6)

        try:
            return try_extract(response, segments, block_map)
        except ValueError as ve:
            print(f"Attempt {attempt}/{max_tries}: JSON tags not found. Retrying…")
            last_err = ve
        except json.JSONDecodeError as je:
            print(f"Attempt {attempt}/{max_tries}: Invalid JSON. Retrying…")
            print(
                "Raw payload was:",
                re.search(r"<json-start>(.*?)<json-end>", response, re.DOTALL).group(
                    1,
                ),
            )  # debug
            last_err = je
        except TypeError as te:
            # Unexpected shape (e.g. returned a number or object missing "values")
            print(f"Attempt {attempt}/{max_tries}: {te}")
            last_err = te

    # If we exhaust retries, re-raise the last exception
    raise last_err


def parse_entity_values(
    segments: List[Dict[str, Any]],
    raw_vals: List[str],
    block_map: Dict[str, Dict[str, Any]],
    fuzzy_thresh: float = 0.9,
) -> List[Dict[str, Any]]:
    """
    segments: [{'token','text','word_ids',…},…]
    raw_vals: e.g. ["Foo Company\tSuite 100 (123|456)", …]
    block_map: Textract Id→block (WORD blocks included)
    """
    R = re.compile(r"(.+?)\s\((\d+\|\d+)\)")  # split into (T, token)
    M = {s["token"]: s for s in segments}
    results = []

    for ev in raw_vals:
        parts = R.findall(ev)
        if not parts:
            continue

        all_boxes: List[Tuple[float, float, float, float]] = []
        ok = True

        for text_part, tok in parts:

            seg = M.get(tok)
            if seg is None or not all(
                any(
                    jaro_similarity(word, normalized_word) > fuzzy_thresh
                    for normalized_word in normalize_text(seg["text"]).split()
                )
                for word in normalize_text(text_part).split()
                # if not word.isdigit() 
            ):
                # Handle Hallucination (layout token id is not present or Extracted Entity Text Not found in the segment)
                ok = False
                break
            
            # ground each word in text_part
            for tw in normalize_text(text_part).split():
                best_bb = None
                best_score = 0.0
                for wid in seg["word_ids"]:
                    wb = block_map.get(wid)
                    if not wb:
                        continue
                    wb_words = normalize_text(wb["Text"]).split()
                    wb_score = 0.0
                    for wb_txt in wb_words:
                        score = textdistance.jaro(wb_txt, tw)
                        if score >= fuzzy_thresh and score > best_score:
                            bb = wb["Geometry"]["BoundingBox"]
                            best_bb = (
                                bb["Left"],
                                bb["Top"],
                                bb["Left"] + bb["Width"],
                                bb["Top"] + bb["Height"],
                            )
                            wb_score+= score
                    if wb_score > 0.0:
                        best_score = wb_score / len(wb_words)
                if best_bb is None:
                    ok = False
                    break
                all_boxes.append(best_bb)
            if not ok:
                break

        if not ok:
            continue

        value = "\t".join(text for text, _ in parts)
        bbox = union_word_boxes(all_boxes)
        results.append({"value": value, "bbox": bbox})

    return results


def load_textract_pages(path: str) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """
    Load one Textract‐style JSON file whose top level is [doc_id, page_obj],
    then group all Blocks by their 'Page' number and return a list of
    (doc_page_id, blocks_for_that_page).

    e.g. if doc_id="foo" and page numbers are 1 and 2, returns:
      [("foo_page1", [ ...blocks with Page==1...]),
       ("foo_page2", [ ...blocks with Page==2...])]
    """
    with open(path, "r", encoding="utf-8") as f:
        arr = json.load(f)
    doc_id = arr[0]

    # group by the Block.Page field
    pages = defaultdict(list)
    for page_obj in arr[1:]:
        blocks = page_obj.get("Blocks", [])
        for b in blocks:
            p = b.get("Page", 1)
            pages[p].append(b)

    # emit one entry per page
    out = []
    for pnum in sorted(pages):
        page_id = f"{doc_id}_page{pnum}"
        out.append((page_id, pages[pnum]))
    return out


def jaro_similarity(word1, word2):
    return textdistance.jaro(word1, word2)


def union_word_boxes(
    boxes: List[Tuple[float, float, float, float]],
) -> Tuple[float, float, float, float]:
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return x0, y0, x1, y1


def extract_votes(
    file_path: str,
    include_pages: List[int],
    model: str,
    out_dir: str = "per_page_votes",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]

    for page_id, blocks in load_textract_pages(file_path):

        # 1) reconstruct segments
        segments, block_map = extract_line_segments(blocks)

        pageno = int(page_id.split("page", 1)[1])

        # 2) Exclude the page
        if pageno not in include_pages:
            continue

        # 3) Build annotator votes
        rows: List[Tuple[str, str, str, int]] = []
        try:
            for annotator, template in enumerate(const.PROMPT_TEMPLATES):
                for pii_type in const.PII_TYPES:
                    prompt = build_bbox_prompt(
                        pii_type=pii_type, segments=segments, template=template
                    )
                    entities = extract_with_retries(
                        prompt=prompt,
                        segments=segments,
                        block_map=block_map,
                        max_tries=3,
                        model=model,
                    )
                    for item in entities:
                        rows.append(
                            (page_id, pii_type, item["value"], item["bbox"], annotator)
                        )

            # Save this page's votes to its own csv
            if rows:
                csv_file_name = os.path.basename(page_id)
                df_page = pd.DataFrame(
                    rows, columns=["id", "pii_type", "value", "bbox", "annotator"]
                )
                csv_name = os.path.join(out_dir, f"votes_{csv_file_name}.csv")
                df_page.to_csv(csv_name, index=False)
                print(f"Saved {len(rows)} votes to {csv_name}")

        except Exception as e:
            print(f"❌ Failed to extract {pii_type} after retries:", str(e))
            print(f"Skipping file {page_id}")
            continue
