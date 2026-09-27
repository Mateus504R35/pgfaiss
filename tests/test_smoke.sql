-- test_smoke.sql
-- Teste pequeno para executar ANTES do Deep1B.
-- Requer a extensão faiss_pg já instalada/atualizada.

DROP TABLE IF EXISTS pgfaiss_smoke CASCADE;
CREATE TABLE pgfaiss_smoke (
    id bigint PRIMARY KEY,
    embedding real[] NOT NULL
);

INSERT INTO pgfaiss_smoke(id, embedding) VALUES
    (0, ARRAY[0.0, 0.0, 0.0]::real[]),
    (1, ARRAY[1.0, 0.0, 0.0]::real[]),
    (2, ARRAY[0.0, 1.0, 0.0]::real[]),
    (3, ARRAY[0.0, 0.0, 1.0]::real[]),
    (4, ARRAY[1.0, 1.0, 1.0]::real[]);

SELECT faiss_build_index(
    'smoke_flat',
    'pgfaiss_smoke'::regclass,
    'id',
    'embedding',
    'l2',
    'flat',
    false,
    '{"indexDir":"/tmp/pgfaiss_smoke_indexes","fetchBatchSize":2}'
);

-- O primeiro resultado deve ser id=0 com distance=0.
SELECT *
FROM faiss_search(
    'smoke_flat',
    ARRAY[0.0, 0.0, 0.0]::real[],
    3
);

-- Duas queries 3D em uma chamada. query_no é 0-based.
SELECT *
FROM faiss_search_batch(
    'smoke_flat',
    ARRAY[
        0.0, 0.0, 0.0,
        1.0, 1.0, 1.0
    ]::real[],
    2,
    2
);

SELECT
    name,
    dim,
    metric,
    index_type,
    ntotal,
    training_rows,
    build_ms,
    dirty,
    source_table_name
FROM faiss_indexes
WHERE name = 'smoke_flat';

-- Verifica dirty tracking sem tentar buscar depois (a busca deve recusar dirty=true).
UPDATE pgfaiss_smoke
   SET embedding = ARRAY[0.1, 0.0, 0.0]::real[]
 WHERE id = 0;

SELECT name, dirty
FROM faiss_indexes
WHERE name = 'smoke_flat';
