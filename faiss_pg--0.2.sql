-- faiss_pg--0.2.sql
\echo Use "CREATE EXTENSION faiss_pg" to load this file. \quit

-- -----------------------------------------------------------------------------
-- API legada (mantida por compatibilidade)
-- -----------------------------------------------------------------------------

CREATE FUNCTION faiss_knn_l2(query real[], data real[], n integer, d integer, k integer)
RETURNS int[]
AS 'MODULE_PATHNAME', 'faiss_knn_l2'
LANGUAGE C STRICT;

CREATE FUNCTION faiss_knn_l2_table(query real[], k integer)
RETURNS int[]
AS 'MODULE_PATHNAME', 'faiss_knn_l2_table'
LANGUAGE C STRICT;

-- -----------------------------------------------------------------------------
-- Catálogo de índices persistidos
-- -----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS faiss_indexes (
    name text PRIMARY KEY,
    dim integer NOT NULL,
    metric text NOT NULL DEFAULT 'cosine',
    index_type text NOT NULL DEFAULT 'flat',
    normalize_vectors boolean NOT NULL DEFAULT true,
    ntotal bigint NOT NULL DEFAULT 0,
    index_path text NOT NULL,
    params text NOT NULL DEFAULT '{}',

    -- Origem do índice (usada para auditoria e dirty tracking)
    source_table_oid oid,
    source_table_name text,
    id_column text,
    embedding_column text,

    -- Origem/amostra de treinamento (IVF)
    training_source_table_oid oid,
    training_source_table_name text,

    -- Métricas de construção úteis nos experimentos
    training_rows bigint NOT NULL DEFAULT 0,
    build_ms double precision,

    updated_at timestamptz NOT NULL DEFAULT now(),
    dirty boolean NOT NULL DEFAULT false
);

-- Marca todos os índices ligados à tabela modificada como desatualizados.
CREATE OR REPLACE FUNCTION faiss_mark_dirty()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE faiss_indexes
       SET dirty = true
     WHERE source_table_oid = TG_RELID;
    RETURN NULL;
END;
$$;

-- Instala um único trigger por tabela. Esse trigger serve para todos os índices
-- Faiss construídos sobre a mesma tabela.
CREATE OR REPLACE FUNCTION faiss_ensure_dirty_trigger(table_oid oid)
RETURNS void
LANGUAGE plpgsql
AS $$
DECLARE
    qualified_name text;
BEGIN
    SELECT format('%I.%I', n.nspname, c.relname)
      INTO qualified_name
      FROM pg_class c
      JOIN pg_namespace n ON n.oid = c.relnamespace
     WHERE c.oid = table_oid;

    IF qualified_name IS NULL THEN
        RAISE EXCEPTION 'source relation with oid % does not exist', table_oid;
    END IF;

    EXECUTE format('DROP TRIGGER IF EXISTS trg_faiss_pg_dirty ON %s', qualified_name);
    EXECUTE format(
        'CREATE TRIGGER trg_faiss_pg_dirty '
        'AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON %s '
        'FOR EACH STATEMENT EXECUTE FUNCTION faiss_mark_dirty()',
        qualified_name
    );
END;
$$;

-- -----------------------------------------------------------------------------
-- Construção
-- -----------------------------------------------------------------------------

-- Compatibilidade: faiss_items(id, embedding), flat/l2.
CREATE FUNCTION faiss_build_index(index_name text)
RETURNS void
AS 'MODULE_PATHNAME', 'faiss_build_index'
LANGUAGE C STRICT;

-- params disponíveis:
-- Gerais:
--   indexDir       caminho onde o .faiss será salvo
--   fetchBatchSize quantidade de linhas lidas/adicionadas por batch (default 10000)
-- HNSW:
--   M, efConstruction, efSearch
-- IVF Flat:
--   nlist, nprobe, trainSize, trainSeed
--   trainingTable, trainingIdColumn, trainingEmbeddingColumn (opcionais)
CREATE FUNCTION faiss_build_index(index_name text,
                                  table_name regclass,
                                  id_column name,
                                  embedding_column name,
                                  metric text,
                                  index_type text,
                                  normalize_vectors boolean,
                                  params text)
RETURNS void
AS 'MODULE_PATHNAME', 'faiss_build_index'
LANGUAGE C STRICT;

-- -----------------------------------------------------------------------------
-- Busca
--
-- Observação: a coluna ainda se chama "distance" por compatibilidade.
-- Em metric='l2', Faiss retorna L2 ao quadrado. Em IP/cosine, o valor é uma
-- similaridade (maior é melhor), não uma distância Euclidiana.
-- -----------------------------------------------------------------------------

CREATE FUNCTION faiss_search(index_name text, query real[], k integer)
RETURNS TABLE(id bigint, distance real)
AS 'MODULE_PATHNAME', 'faiss_search'
LANGUAGE C STRICT;

CREATE FUNCTION faiss_search(index_name text,
                             query real[],
                             k integer,
                             search_params text)
RETURNS TABLE(id bigint, distance real)
AS 'MODULE_PATHNAME', 'faiss_search'
LANGUAGE C STRICT;

-- queries é um real[] achatado contendo nq vetores consecutivos.
-- query_no é 0-based.
CREATE FUNCTION faiss_search_batch(index_name text,
                                   queries real[],
                                   nq integer,
                                   k integer)
RETURNS TABLE(query_no integer, id bigint, distance real)
AS 'MODULE_PATHNAME', 'faiss_search_batch'
LANGUAGE C STRICT;

CREATE FUNCTION faiss_search_batch(index_name text,
                                   queries real[],
                                   nq integer,
                                   k integer,
                                   search_params text)
RETURNS TABLE(query_no integer, id bigint, distance real)
AS 'MODULE_PATHNAME', 'faiss_search_batch'
LANGUAGE C STRICT;

-- -----------------------------------------------------------------------------
-- Cache por backend PostgreSQL
-- -----------------------------------------------------------------------------

CREATE FUNCTION faiss_clear_cache(index_name text)
RETURNS void
AS 'MODULE_PATHNAME', 'faiss_clear_cache'
LANGUAGE C STRICT;

CREATE FUNCTION faiss_clear_all_cache()
RETURNS void
AS 'MODULE_PATHNAME', 'faiss_clear_all_cache'
LANGUAGE C;
