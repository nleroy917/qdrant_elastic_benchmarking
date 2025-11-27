import polars as pl

from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from umap import UMAP

MODEL = "Snowflake/snowflake-arctic-embed-xs"
model = SentenceTransformer(MODEL, device="mps")

# load the women's clothing ecommerce reviews dataset
ds = load_dataset("saattrupdan/womens-clothing-ecommerce-reviews")

# convert to polars DataFrame
df = ds["train"].to_polars()

# Rename review_text to text and keep all other columns, filter out empty reviews
df = df.with_columns(
    pl.col("review_text").alias("text")
).fill_null("")

df_final = df.filter(pl.col("text") != "")

# compute embeddings
texts = df_final["text"].to_list()
for i, t in enumerate(texts):
    if t is None or len(t) == 0:
        print(f"Empty text found at index {i}: {repr(t)}")
        raise ValueError(f"Text at index {i} is empty after processing.")

embeddings = model.encode(texts, show_progress_bar=True)

# attach metadata to embeddings
df_final = df_final.with_columns(
    pl.Series("embedding", embeddings.tolist())
)

# perform umap
umap_model = UMAP(n_components=2, random_state=42)
embeddings_2d = umap_model.fit_transform(embeddings)

df_final = df_final.with_columns(
    umap1=pl.Series([e[0] for e in embeddings_2d]),
    umap2=pl.Series([e[1] for e in embeddings_2d]),
)

# save to parquet
df_final.write_parquet("data/ecommerce_products_with_embeddings.parquet")

