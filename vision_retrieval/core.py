import base64
import io
import os
from typing import Optional

import lancedb
import numpy as np
import PIL
import PIL.Image
import requests
import torch
from colpali_engine.models.paligemma_colbert_architecture import ColPali
from colpali_engine.trainer.retrieval_evaluator import CustomEvaluator
from colpali_engine.utils.colpali_processing_utils import (
    process_images,
    process_queries,
)
from colpali_engine.utils.image_utils import get_base64_image
from pdf2image import convert_from_path
from pypdf import PdfReader
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


def base64_to_pil(base64_str: str) -> PIL.Image.Image:
    if base64_str.startswith("data:image"):
        base64_str = base64_str.split(",")[1]
    image_data = base64.b64decode(base64_str)
    image = PIL.Image.open(io.BytesIO(image_data))
    return image


def download_pdf(url: str, save_directory: str = "."):
    response = requests.get(url)
    if response.status_code == 200:
        # Check for Content-Disposition header to get the filename
        if "Content-Disposition" in response.headers:
            # Extract filename from header if available
            filename = response.headers.get("Content-Disposition").split("filename=")[-1].strip('"')
        else:
            # Fallback: Use the last part of the URL as filename
            filename = os.path.basename(url)
            # Ensure the file has a .pdf extension
            if not filename.endswith(".pdf"):
                filename += ".pdf"

        # Save the file to the specified directory
        file_path = os.path.join(save_directory, filename)
        with open(file_path, "wb") as file:
            file.write(response.content)

        print(f"PDF downloaded and saved as {file_path}")
        return file_path
    else:
        raise Exception(f"Failed to download PDF: Status code {response.status_code}")


def get_pdf_images(pdf_path):
    reader = PdfReader(pdf_path)
    page_texts = []
    for page_number in range(len(reader.pages)):
        page = reader.pages[page_number]
        text = page.extract_text()
        page_texts.append(text)

    images = convert_from_path(pdf_path)
    assert len(images) == len(page_texts)
    return (images, page_texts)


def get_model_colpali(base_model_id: Optional[str] = None):
    model_name = "vidore/colpali"
    if base_model_id is None:
        base_model_id = "google/paligemma-3b-mix-448"
    model = ColPali.from_pretrained(base_model_id, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    model.load_adapter(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor


def get_pdf_embedding(pdf_path: str, model, processor):
    page_images, page_texts = get_pdf_images(pdf_path=pdf_path)
    page_embeddings = []
    dataloader = DataLoader(
        page_images,
        batch_size=2,
        shuffle=False,
        collate_fn=lambda x: process_images(processor, x),
    )

    for batch_doc in tqdm(dataloader):
        with torch.no_grad():
            batch_doc = {k: v.to(model.device) for k, v in batch_doc.items()}
            embeddings_doc = model(**batch_doc)
            page_embeddings.extend(list(torch.unbind(embeddings_doc.to("cpu"))))

    document = {
        "name": pdf_path,
        "page_images": page_images,
        "page_texts": page_texts,
        "page_embeddings": page_embeddings,
    }
    return document


def get_query_embedding(query: str, model, processor):
    dummy_image = PIL.Image.new("RGB", (448, 448), (255, 255, 255))
    dataloader = DataLoader(
        [query],
        batch_size=4,
        shuffle=False,
        collate_fn=lambda x: process_queries(processor, x, dummy_image),
    )

    qs = []
    for batch_query in dataloader:
        with torch.no_grad():
            batch_query = {k: v.to(model.device) for k, v in batch_query.items()}
            embeddings_query = model(**batch_query)
        qs.extend(list(torch.unbind(embeddings_query.to("cpu"))))

    q = {"query": query, "embeddings": qs[0]}
    return q


def embedd_docs(docs_path, model, processor):
    docs_to_store_pages = []

    for pdf_path in docs_path:
        print(pdf_path)
        pdf_doc = get_pdf_embedding(pdf_path=pdf_path, model=model, processor=processor)
        for page_idx in range(len(pdf_doc["page_images"])):
            docs_to_store_pages.append(
                {
                    "name": pdf_doc["name"],
                    "page_idx": page_idx,
                    "page_image": pdf_doc["page_images"][page_idx],
                    "page_text": pdf_doc["page_texts"][page_idx],
                    "page_embedding": pdf_doc["page_embeddings"][page_idx],
                }
            )

    return docs_to_store_pages


def create_db(docs_storage, table_name: str = "demo", db_path: str = "lancedb"):
    db = lancedb.connect(db_path)

    data = []
    for x in docs_storage:
        sample = {
            "name": x["name"],
            "page_texts": x["page_text"],
            "image": get_base64_image(x["page_image"]),
            "page_idx": x["page_idx"],
        }
        patch = {f"patch_{idx}": x["page_embedding"][idx].float().numpy() for idx in range(len(x["page_embedding"]))}
        sample.update(patch)
        data.append(sample)

    table = db.create_table(table_name, data, mode="overwrite")
    return table


def search(query, table_name: str, model, processor, db_path: str = "lancedb"):
    qs = get_query_embedding(query=query, model=model, processor=processor)
    db = lancedb.connect(db_path)
    table = db.open_table(table_name)
    r = table.search().limit(1000).to_list()

    def marge_patch(record):
        patches = np.array([record[f"patch_{idx}"] for idx in range(1030)])
        page_embeddings = torch.from_numpy(patches).to(torch.bfloat16)
        return page_embeddings

    all_pages_embeddings = [marge_patch(x) for x in r]
    retriever_evaluator = CustomEvaluator(is_multi_vector=True)
    scores = retriever_evaluator.evaluate_colbert([qs["embeddings"]], all_pages_embeddings)
    # TODO: return top k images
    page = r[scores.argmax(axis=1)]
    pil_image = base64_to_pil(page["image"])
    meta = {"name": page["name"], "page_idx": page["page_idx"]}
    return pil_image, meta


def get_model_phi_vision(model_id: Optional[str] = None):
    if model_id is None:
        model_id = "microsoft/Phi-3.5-vision-instruct"
    # Note: set _attn_implementation='eager' if you don't have flash_attn installed
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        trust_remote_code=True,
        torch_dtype="auto",
        # _attn_implementation='flash_attention_2'
        _attn_implementation="eager",
    )
    # for best performance, use num_crops=4 for multi-frame, num_crops=16 for single-frame.
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, num_crops=4)
    return model, processor


def run_vision_inference(input_images: PIL.Image, prompt: str, model, processor):
    images = []
    placeholder = ""

    # Note: if OOM, you might consider reduce number of frames in this example.
    for i in range(len(input_images)):
        images.append(input_images[i])
        placeholder += f"<|image_{i + 1}|>\n"

    messages = [
        {"role": "user", "content": f"{placeholder} {prompt}"},
    ]

    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = processor(prompt, images, return_tensors="pt").to("cuda:0")

    generation_args = {
        "max_new_tokens": 512,
        "temperature": 0.2,
        "do_sample": True,
    }

    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    # remove input tokens
    generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return response
