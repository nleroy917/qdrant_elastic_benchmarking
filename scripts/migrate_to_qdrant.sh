#!/bin/bash

docker run --net=host --rm -it registry.cloud.qdrant.io/library/qdrant-migration elasticsearch \
    --elasticsearch.url 'http://localhost:9200' \
    --elasticsearch.insecure-skip-verify \
    --elasticsearch.index 'bench_write' \
    --elasticsearch.api-key $ES_LOCAL_API_KEY \
    --qdrant.url 'http://localhost:6334' \
    --qdrant.collection 'bench_write' \
    --migration.batch-size 64
    