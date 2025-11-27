#!/bin/bash

docker run --net=host --rm -it registry.cloud.qdrant.io/library/qdrant-migration elasticsearch \
    --elasticsearch.url 'http://localhost:9200' \
    --elasticsearch.insecure-skip-verify \
    --elasticsearch.index 'womens_clothing_reviews' \
    --elasticsearch.api-key $ES_LOCAL_API_KEY \
    --qdrant.url 'http://localhost:6334' \
    --qdrant.collection 'womens_clothing_reviews' \
    --migration.batch-size 64
    