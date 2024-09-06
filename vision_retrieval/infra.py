import os
from typing import List

import modal


def download_model_to_image(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)
    snapshot_download(
        model_name,
        local_dir=model_dir,
        token=os.environ["HF_TOKEN"],
        ignore_patterns=["*.pt", "*.gguf"],
    )
    move_cache()


image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "poppler-utils")
    .pip_install(
        "numpy==1.24.4",
        "Pillow==10.3.0",
        "Requests==2.31.0",
        "torch==2.3.0",
        "torchvision==0.18.0",
        "transformers==4.43.0",
        "accelerate==0.30.0",
        "git+https://github.com/ManuelFay/colpali@9413418c110da49b25ac2dae2c32b8fc067ff332",
        "pdf2image==1.17.0",
        "pypdf==4.3.1",
        "lancedb==0.12.0",
        "hf-transfer==0.1.4",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
        timeout=60 * 20,
        kwargs={"model_dir": "/model-phi-3.5-vision-instruct", "model_name": "microsoft/Phi-3.5-vision-instruct"},
    )
    .run_function(
        download_model_to_image,
        secrets=[modal.Secret.from_name("huggingface-secret", required_keys=["HF_TOKEN"])],
        timeout=60 * 20,
        kwargs={"model_dir": "/model-paligemma-3b-mix-448", "model_name": "google/paligemma-3b-mix-448"},
    )
)


app = modal.App("vision_retrieval", image=image)
mounts = [modal.Mount.from_local_python_packages("vision_retrieval", "vision_retrieval")]


@app.cls(
    gpu="a10g",
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("aws-secret")],
    mounts=mounts,
    container_idle_timeout=300,
    keep_warm=1,
    allow_concurrent_inputs=4,
)
class VisionRAG:
    @modal.enter()
    def load(self):
        import time

        start = time.monotonic()

        from vision_retrieval.core import get_model_colpali, get_model_phi_vision

        model_colpali, processor_colpali = get_model_colpali(base_model_id="/model-paligemma-3b-mix-448")
        model_phi_vision, processor_phi_vision = get_model_phi_vision(model_id="/model-phi-3.5-vision-instruct")

        self.model_colpali = model_colpali
        self.processor_colpali = processor_colpali

        self.model_phi_vision = model_phi_vision
        self.processor_phi_vision = processor_phi_vision

        end = time.monotonic()
        print(f"load takes {end - start}")

    @modal.method()
    def ask_llm_example(self):
        import PIL.Image
        import requests

        from vision_retrieval.core import run_vision_inference

        images = []
        for i in range(1, 2):
            url = f"https://image.slidesharecdn.com/azureintroduction-191206101932/75/Introduction-to-Microsoft-Azure-Cloud-{i}-2048.jpg"
            images.append(PIL.Image.open(requests.get(url, stream=True).raw))

        prompt = "Summarize the deck of slides."
        response = run_vision_inference(
            input_images=images, prompt=prompt, model=self.model_phi_vision, processor=self.processor_phi_vision
        )
        print(f"response = {response}")

        print(response)

    @modal.method()
    def ingest_data(self, pdf_urls: List[str], table_name: str, db_path: str):
        print("1. Downloads PDFs")
        from tqdm import tqdm

        from vision_retrieval.core import create_db, download_pdf, embedd_docs
        pdfs = []
        for pdf_url in tqdm(pdf_urls):
            pdf_file_name = download_pdf(url=pdf_url)
            pdfs.append(pdf_file_name)
        print(f"result pdfs = {pdfs}")

        print("2. Generating embeddings")
        docs_storage = embedd_docs(docs_path=pdfs, model=self.model_colpali, processor=self.processor_colpali)

        print(f"result docs = {len(docs_storage)}")

        print("3. Build vectorDB")
        create_db(docs_storage=docs_storage, table_name=table_name, db_path=db_path)
        print("Done!")

    @modal.method()
    def query_data(self, user_query: str, table_name: str, db_path: str):
        from vision_retrieval.core import run_vision_inference, search

        print("1. Search relevant images")
        search_results = search(
            query=user_query,
            table_name=table_name,
            db_path=db_path,
            processor=self.processor_colpali,
            model=self.model_colpali,
        )
        print(f"result most relevant {search_results[0]}")

        print("2. Build prompt")
        # https://cookbook.openai.com/examples/custom_image_embedding_search#user-querying-the-most-similar-image
        prompt = f"""
        Below is a user query, I want you to answer the query using images provided.
        user query:
        {user_query}
        """

        print("3. Query LLM with prompt and relavent images")
        response = run_vision_inference(
            input_images=[search_results[0]['pil_image']], prompt=prompt, model=self.model_phi_vision, processor=self.processor_phi_vision
        )
        print(f"response = {response}")
        return {"response": response, "page": search_results[0]['page_idx'] + 1, "pdf_name": search_results[0]['name']}


@app.local_entrypoint()
def main():
    vision_rag = VisionRAG()
    # pdf_urls = ["https://vision-retrieval.s3.amazonaws.com/docs/budget-2024.pdf", "https://vision-retrieval.s3.amazonaws.com/docs/CartaVCFundPerformanceQ12024.pdf", "https://vision-retrieval.s3.amazonaws.com/docs/InfraRedReport.pdf"]
    pdf_urls = ["https://vision-retrieval.s3.amazonaws.com/docs/InfraRedReport.pdf"]
    db_path = "s3://vision-retrieval/storage"
    table_name = "table-test"

    vision_rag.ingest_data.remote(pdf_urls=pdf_urls, table_name=table_name, db_path=db_path)

    user_query = "How does training costs change over time?"
    vision_rag.query_data.remote(user_query=user_query, table_name=table_name, db_path=db_path)


def python_main():
    VisionRAG = modal.Cls.lookup("vision_retrieval", "VisionRAG")
    vision_rag = VisionRAG()

    pdf_urls = ["https://vision-retrieval.s3.amazonaws.com/docs/InfraRedReport.pdf"]
    db_path = "s3://vision-retrieval/storage"
    table_name = "table-test"

    vision_rag.ingest_data.remote(pdf_urls=pdf_urls, table_name=table_name, db_path=db_path)

    user_query = "How does inference costs change over time?"
    user_query = "Top companies in observability space?"
    vision_rag.query_data.remote(user_query=user_query, table_name=table_name, db_path=db_path)

if __name__ == '__main__':
    python_main()
