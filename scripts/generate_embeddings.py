import polars as pl

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from umap import UMAP

MODEL = "Snowflake/snowflake-arctic-embed-xs"
model = SentenceTransformer(MODEL, device="mps")

# load the dataset
ds = load_dataset("DenCT/amazon_products_23")

# convert to polars DataFrame
df = ds["train"].to_polars()

# compute embeddings
texts = df["description"].to_list()
for i, t in enumerate(texts):
    if t is None or len(t) == 0:
        print(f"Empty text found at index {i}: {repr(t)}")
        raise ValueError(f"Text at index {i} is empty after processing.")

embeddings = model.encode(texts, show_progress_bar=True)
    
# attach metadata to embeddings
df = df.with_columns(
    pl.Series("embedding", embeddings.tolist())
)

# perform umap
umap_model = UMAP(n_components=2, random_state=42)
embeddings_2d = umap_model.fit_transform(embeddings)

df = df.with_columns(
    umap1=pl.Series([e[0] for e in embeddings_2d]),
    umap2=pl.Series([e[1] for e in embeddings_2d]),
)

# save to parquet
df.write_parquet("data/ecommerce_products_with_embeddings.parquet")

