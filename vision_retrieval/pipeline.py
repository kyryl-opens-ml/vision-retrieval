from dagster import asset, MetadataValue, AssetExecutionContext
import modal
import lancedb

import nest_asyncio
import asyncio

# Allow nested event loops
nest_asyncio.apply()

@asset(group_name="ingest", compute_kind="python")
def pdf_corpus(context: AssetExecutionContext):
    pdf_urls = [
        "https://vision-retrieval.s3.amazonaws.com/docs/InfraRedReport.pdf",
    ]
    context.add_output_metadata(
        {
            "len": MetadataValue.int(len(pdf_urls)),
            "sample": MetadataValue.json(pdf_urls),
        }
    )

    return pdf_urls


@asset(group_name="ingest", compute_kind="modal-lab")
def pdf_embeddings_table(context: AssetExecutionContext, pdf_corpus):
    VisionRAG = modal.Cls.lookup("vision_retrieval", "VisionRAG")
    vision_rag = VisionRAG()

    # TODO: move to configs
    db_path = "s3://vision-retrieval/storage"
    table_name = "dagster-table"

    vision_rag.ingest_data.remote(pdf_urls=pdf_corpus, table_name=table_name, db_path=db_path)

    db = lancedb.connect(db_path)
    table_names = db.table_names()
    t = db.open_table(table_name)
    schema = t.schema

    context.add_output_metadata(
        {
            "table_names": MetadataValue.json(table_names),
            "table_schema": MetadataValue.md(schema.to_string()),
        }
    )

    return table_name


@asset(group_name="query", compute_kind="python")
def query_samples(context: AssetExecutionContext):
    query_samples_ = ["How does inference costs change over time?", "Top companies in observability space?"]
    context.add_output_metadata(
        {
            "len": MetadataValue.int(len(query_samples_)),
            "sample": MetadataValue.json(query_samples_),
        }
    )
    return query_samples_


@asset(group_name="query", compute_kind="python")
def query_results(context: AssetExecutionContext, query_samples, pdf_embeddings_table):
    VisionRAG = modal.Cls.lookup("vision_retrieval", "VisionRAG")
    vision_rag = VisionRAG()

    # TODO: move to configs
    db_path = "s3://vision-retrieval/storage"
    
    data = []
    for q in query_samples:
        result = vision_rag.query_data.remote(user_query=q, table_name=pdf_embeddings_table, db_path=db_path)
        result['query'] = q
        data.append(result)

    context.add_output_metadata(
        {
            "data": MetadataValue.json(data),
        }
    )

