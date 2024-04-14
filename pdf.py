from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import (StorageContext,
                              VectorStoreIndex,
                              load_index_from_storage)
from llama_index.readers.file import PDFReader


load_dotenv()


def get_index(data, index_name):
    index = None
    if not Path(index_name).exists():
        print(f'Building index {index_name}...')
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name))

    return index


pdf_path = Path("data/Brasil.pdf")
brasil_pdf = PDFReader().load_data(file=pdf_path)
brasil_index = get_index(brasil_pdf, 'brasil')
brasil_engine = brasil_index.as_query_engine()
