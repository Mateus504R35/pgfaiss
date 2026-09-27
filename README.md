# pgfaiss

`pgfaiss` is a PostgreSQL extension that integrates the C++ [Faiss](https://github.com/facebookresearch/faiss) library with PostgreSQL tables for exact and approximate nearest-neighbor search over dense vectors.

The PostgreSQL extension itself is named **`faiss_pg`**. It builds Faiss indexes from `real[]` columns, persists the indexes as `.faiss` files, stores their metadata in PostgreSQL, and exposes SQL functions for index construction and search.

This repository was developed in the context of an undergraduate scientific research project at the Federal University of Uberlândia (UFU) involving similarity search and indexing of high-dimensional data.

## Features

- Exact search with **Flat** indexes.
- Approximate search with **HNSW**.
- Approximate search with **IVF-Flat**.
- Distance/similarity modes:
  - L2;
  - inner product;
  - cosine similarity through L2 normalization + inner product.
- Index construction in batches, avoiding a second complete in-memory copy of the dataset.
- Reproducible reservoir sampling for IVF training.
- Optional separate training table for IVF indexes.
- Single-query and batched-query search functions.
- Per-PostgreSQL-backend in-memory cache of loaded Faiss indexes.
- Persistent index metadata in the `faiss_indexes` table.
- Statement-level dirty tracking: changes to the source table mark associated Faiss indexes as stale.
- Configurable search parameters such as HNSW `efSearch` and IVF `nprobe` without mutating the cached index.

## Important design note

`pgfaiss` is not a PostgreSQL index access method and does not currently implement `CREATE INDEX`, an operator class, or a planner-integrated scan.

Instead, the extension:

1. reads vectors from a PostgreSQL table;
2. creates a Faiss index in C++;
3. stores that index in a `.faiss` file;
4. stores metadata about the index in PostgreSQL;
5. performs searches through SQL functions such as `faiss_search()`.

## Repository layout

```text
.
├── third_party/
│   └── faiss/              # Git submodule pinned to Faiss v1.13.1
├── tests/
│   └── test_smoke.sql
├── faiss_pg.cpp            # PostgreSQL/Faiss integration
├── faiss_pg--0.2.sql       # SQL objects exposed by extension version 0.2
├── faiss_pg.control        # PostgreSQL extension metadata
├── Makefile                # PGXS build file
├── .gitignore
└── README.md
```

## Faiss dependency

Faiss is included as a Git submodule under `third_party/faiss` and should be pinned to **v1.13.1**.

Clone this repository together with the submodule:

```bash
git clone --recurse-submodules <YOUR_REPOSITORY_URL>
cd pgfaiss
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

## Requirements

The current extension targets a Linux/PostgreSQL build environment and uses the Faiss CPU library.

Main requirements:

- PostgreSQL and PostgreSQL server development headers;
- a C++17 compiler;
- CMake;
- OpenMP support;
- BLAS/LAPACK implementation;
- Faiss v1.13.1.

Example for Ubuntu with PostgreSQL 16:

```bash
sudo apt update
sudo apt install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    postgresql-server-dev-16
```

If another PostgreSQL major version is being used, install the matching `postgresql-server-dev-*` package.

## Building Faiss v1.13.1

The submodule provides the exact Faiss source revision used by this project.

Configure a CPU-only shared-library build:

```bash
cmake -S third_party/faiss -B third_party/faiss/build \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBUILD_TESTING=OFF \
    -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local
```

Compile Faiss:

```bash
make -C third_party/faiss/build -j4 faiss
```

Install the library and headers:

```bash
sudo make -C third_party/faiss/build install
sudo ldconfig
```

The default `Makefile` for `faiss_pg` expects Faiss under `/usr/local`. A different prefix can be supplied with `FAISS_PREFIX`, for example:

```bash
make FAISS_PREFIX=/opt/faiss
```

## Building the PostgreSQL extension

If only one PostgreSQL development version is installed and `pg_config` points to the intended server version:

```bash
make clean
make
sudo make install
```

When multiple PostgreSQL versions are installed, explicitly select the desired `pg_config`. For PostgreSQL 16, for example:

```bash
make clean
make PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config
sudo make PG_CONFIG=/usr/lib/postgresql/16/bin/pg_config install
```

Check which version is selected with:

```bash
pg_config --version
```

## Creating the extension

Connect to the target database and run:

```sql
CREATE EXTENSION faiss_pg;
```

The installation creates the `faiss_indexes` metadata table and the SQL functions exposed by the extension.

## Source-table requirements

The general index builder expects:

- an ID column of type `integer` (`int4`) or `bigint` (`int8`);
- an embedding column of type `real[]` (`float4[]`);
- one-dimensional embeddings;
- no NULL elements inside embeddings;
- non-NULL IDs and embeddings;
- the same dimensionality for every vector in the indexed table.

Example table:

```sql
CREATE TABLE items (
    id bigint PRIMARY KEY,
    embedding real[] NOT NULL
);
```

## Index storage

By default, index files are stored under:

```text
/var/lib/postgresql/faiss_indexes
```

The PostgreSQL operating-system user must be able to create/write this directory. It can be created manually with:

```bash
sudo install -d -o postgres -g postgres -m 700 /var/lib/postgresql/faiss_indexes
```

A different directory can be supplied through the `indexDir` build parameter.

## Building indexes

The general build function is:

```sql
faiss_build_index(
    index_name text,
    table_name regclass,
    id_column name,
    embedding_column name,
    metric text,
    index_type text,
    normalize_vectors boolean,
    params text
)
```

### Flat

Flat performs exact search.

```sql
SELECT faiss_build_index(
    'items_flat_l2',
    'public.items'::regclass,
    'id'::name,
    'embedding'::name,
    'l2',
    'flat',
    false,
    '{"fetchBatchSize":10000}'
);
```

### HNSW

```sql
SELECT faiss_build_index(
    'items_hnsw',
    'public.items'::regclass,
    'id'::name,
    'embedding'::name,
    'cosine',
    'hnsw32',
    true,
    '{"M":32,"efConstruction":200,"efSearch":128,"fetchBatchSize":10000}'
);
```

`hnsw32` encodes `M = 32`. When the index type contains the numeric suffix, that value takes precedence over the `M` field in `params`.

### IVF-Flat

```sql
SELECT faiss_build_index(
    'items_ivfflat',
    'public.items'::regclass,
    'id'::name,
    'embedding'::name,
    'cosine',
    'ivfflat',
    true,
    '{"nlist":1024,"nprobe":32,"trainSize":100000,"trainSeed":42,"fetchBatchSize":10000}'
);
```

IVF training uses reproducible reservoir sampling. By default, the training sample is drawn from the indexed table.

A separate training table can be provided:

```sql
SELECT faiss_build_index(
    'items_ivfflat',
    'public.items'::regclass,
    'id'::name,
    'embedding'::name,
    'cosine',
    'ivfflat',
    true,
    '{
        "nlist":1024,
        "nprobe":32,
        "trainSize":100000,
        "trainSeed":42,
        "trainingTable":"public.items_learn",
        "trainingIdColumn":"id",
        "trainingEmbeddingColumn":"embedding"
    }'
);
```

## Build parameters

The `params` argument is a text value containing JSON-style key/value pairs.

| Scope | Parameter | Meaning |
|---|---|---|
| General | `indexDir` | Directory used to store the `.faiss` file |
| General | `fetchBatchSize` | Number of table rows read/added per batch; default `10000` |
| HNSW | `M` | Number of HNSW graph connections; default `32` |
| HNSW | `efConstruction` | HNSW construction parameter; default `200` |
| HNSW | `efSearch` | Default HNSW search parameter; default `128` |
| IVF | `nlist` | Number of inverted lists; default `4096`, capped at table size |
| IVF | `nprobe` | Default number of lists searched; default `64` |
| IVF | `trainSize` | Number of training vectors |
| IVF | `trainSeed` | Reservoir-sampling seed; default `42` |
| IVF | `trainingTable` | Optional separate relation used for training |
| IVF | `trainingIdColumn` | ID column used to order the training scan |
| IVF | `trainingEmbeddingColumn` | Embedding column in the training table |

For IVF, the default training size is `max(100000, 40 * nlist)`, capped by the number of rows available for training.

## Searching

### Single query

```sql
SELECT *
FROM faiss_search(
    'items_flat_l2',
    ARRAY[0.1, 0.2, 0.3]::real[],
    10
);
```

The result contains:

```text
id | distance
```

The column is named `distance` for compatibility, but its semantics depend on the metric:

- `l2`: Faiss returns **squared L2 distance**; lower is better;
- `ip`: value is inner-product similarity; higher is better;
- `cosine`: value is inner-product similarity over normalized vectors; higher is better.

### HNSW search-time parameters

`efSearch` can be changed per query without changing the cached index:

```sql
SELECT *
FROM faiss_search(
    'items_hnsw',
    ARRAY[0.1, 0.2, 0.3]::real[],
    10,
    '{"efSearch":256}'
);
```

### IVF search-time parameters

```sql
SELECT *
FROM faiss_search(
    'items_ivfflat',
    ARRAY[0.1, 0.2, 0.3]::real[],
    10,
    '{"nprobe":64,"maxCodes":0}'
);
```

## Batch search

`faiss_search_batch()` performs multiple queries in one Faiss call.

The input `queries` is a flattened `real[]` containing `nq` consecutive vectors.

Example with two 3-dimensional queries:

```sql
SELECT *
FROM faiss_search_batch(
    'items_flat_l2',
    ARRAY[
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6
    ]::real[],
    2,
    10
);
```

The result contains:

```text
query_no | id | distance
```

`query_no` is zero-based.

The overload with search parameters is also available:

```sql
SELECT *
FROM faiss_search_batch(
    'items_hnsw',
    ARRAY[
        0.1, 0.2, 0.3,
        0.4, 0.5, 0.6
    ]::real[],
    2,
    10,
    '{"efSearch":256}'
);
```

## Metadata and dirty tracking

Every built index is registered in:

```sql
SELECT * FROM faiss_indexes;
```

The table stores, among other fields:

- index name;
- dimension;
- metric;
- index type;
- normalization setting;
- number of indexed vectors;
- `.faiss` file path;
- source table and columns;
- IVF training source;
- number of training rows;
- build time in milliseconds;
- last update timestamp;
- `dirty` status.

When an indexed source table receives `INSERT`, `UPDATE`, `DELETE`, or `TRUNCATE`, a statement-level trigger marks its associated Faiss indexes as dirty.

A search against a dirty index fails with an error until the index is rebuilt:

```sql
SELECT faiss_build_index(...);
```

This prevents queries from silently using an index that no longer represents the current table contents.

## Cache behavior

Faiss index files are loaded lazily and cached in memory per PostgreSQL backend process.

Clear one cached index:

```sql
SELECT faiss_clear_cache('items_hnsw');
```

Clear all Faiss indexes cached by the current backend:

```sql
SELECT faiss_clear_all_cache();
```

The cache is automatically refreshed after a rebuild because the metadata `updated_at` value changes.

## Concurrency during index construction

During `faiss_build_index()`, the source table is locked in PostgreSQL `SHARE` mode so that the extension builds the Faiss index from a consistent table snapshot without concurrent `INSERT`, `UPDATE`, or `DELETE` operations modifying the indexed data during construction.

For IVF with a separate training table, that relation is also locked while its training sample is collected.

## Smoke test

After installing the extension, run:

```bash
psql -d <DATABASE> -f tests/test_smoke.sql
```

The test:

1. creates a small `real[]` dataset;
2. builds a Flat/L2 index;
3. executes a nearest-neighbor search;
4. inspects `faiss_indexes`;
5. modifies the source table and confirms that dirty tracking is activated.

The final search-after-dirty error is intentionally not executed automatically; the test leaves the metadata visible for inspection.

## Legacy API

Version `0.2` retains the original functions for compatibility:

```sql
faiss_knn_l2(query real[], data real[], n integer, d integer, k integer)
faiss_knn_l2_table(query real[], k integer)
faiss_build_index(index_name text)
```

The one-argument `faiss_build_index()` expects a legacy table named `faiss_items` with `id` and `embedding` columns and builds a Flat/L2 index.

New applications should use the general multi-argument index builder.

## Faiss version

This repository is intended to use **Faiss v1.13.1** through the `third_party/faiss` Git submodule.

Faiss is developed by Meta's Fundamental AI Research group and is distributed under the MIT License. See the Faiss submodule for its own source code, documentation, and license.

## Research context

This implementation was developed to support experimental evaluation of vector similarity-search techniques in PostgreSQL, including exact and approximate nearest-neighbor workloads over high-dimensional datasets.

For benchmark reproducibility, record at least:

- PostgreSQL version;
- Faiss version and submodule commit;
- compiler version;
- BLAS implementation;
- dataset and dimensionality;
- index build parameters;
- search-time parameters;
- hardware configuration.
