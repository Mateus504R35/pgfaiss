// faiss_pg.cpp
// pgfaiss - versão preparada para experimentos em escala (ex.: Deep1B-10M)
//
// Principais mudanças em relação ao protótipo original:
//   * construção Flat/HNSW/IVF em batches, sem manter toda a base duplicada em RAM;
//   * treinamento IVF com amostra reservoir reproduzível;
//   * SPI_freetuptable() a cada batch;
//   * cópia direta de real[] para buffers float32 (sem std::vector temporário por linha);
//   * parâmetros de busca HNSW/IVF locais a cada chamada;
//   * busca em batch;
//   * metadados da tabela/colunas de origem e dirty tracking genérico;
//   * tratamento de exceções do Faiss nos caminhos principais.

extern "C" {
    #include "postgres.h"
    #include "fmgr.h"
    #include "utils/array.h"
    #include "miscadmin.h"
    #include "utils/memutils.h"
    #include "executor/spi.h"
    #include "access/htup_details.h"
    #include "utils/lsyscache.h"
    #include "catalog/pg_type.h"
    #include "catalog/namespace.h"
    #include "utils/builtins.h"
    #include "utils/timestamp.h"
    #include "utils/tuplestore.h"
    #include "funcapi.h"

    PG_MODULE_MAGIC;

    PG_FUNCTION_INFO_V1(faiss_knn_l2);
    Datum faiss_knn_l2(PG_FUNCTION_ARGS);

    PG_FUNCTION_INFO_V1(faiss_knn_l2_table);
    Datum faiss_knn_l2_table(PG_FUNCTION_ARGS);

    PG_FUNCTION_INFO_V1(faiss_build_index);
    Datum faiss_build_index(PG_FUNCTION_ARGS);

    PG_FUNCTION_INFO_V1(faiss_search);
    Datum faiss_search(PG_FUNCTION_ARGS);

    PG_FUNCTION_INFO_V1(faiss_search_batch);
    Datum faiss_search_batch(PG_FUNCTION_ARGS);

    PG_FUNCTION_INFO_V1(faiss_clear_cache);
    Datum faiss_clear_cache(PG_FUNCTION_ARGS);

    PG_FUNCTION_INFO_V1(faiss_clear_all_cache);
    Datum faiss_clear_all_cache(PG_FUNCTION_ARGS);
}

#include <algorithm>
#include <cerrno>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <regex>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>

using std::string;
using std::vector;

static const char* DEFAULT_INDEX_DIR = "/var/lib/postgresql/faiss_indexes";

// -----------------------------------------------------------------------------
// Utilitários de strings / parâmetros
// -----------------------------------------------------------------------------

static string to_lower_copy(string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return (char) std::tolower(c);
    });
    return s;
}

static bool starts_with(const string& value, const string& prefix) {
    return value.rfind(prefix, 0) == 0;
}

static string text_arg_to_string(PG_FUNCTION_ARGS, int argno, const char* fallback = "") {
    if (PG_NARGS() <= argno || PG_ARGISNULL(argno)) {
        return string(fallback);
    }
    text* t = PG_GETARG_TEXT_PP(argno);
    char* c = text_to_cstring(t);
    string out(c);
    pfree(c);
    return out;
}

static int get_json_int_param(const string& params, const string& key, int fallback) {
    try {
        std::regex re("\\\"" + key + "\\\"\\s*:\\s*(-?[0-9]+)");
        std::smatch match;
        if (std::regex_search(params, match, re) && match.size() >= 2) {
            return std::stoi(match[1].str());
        }
    } catch (...) {}
    return fallback;
}

static string get_json_string_param(const string& params, const string& key, const string& fallback) {
    try {
        std::regex re("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]+)\\\"");
        std::smatch match;
        if (std::regex_search(params, match, re) && match.size() >= 2) {
            return match[1].str();
        }
    } catch (...) {}
    return fallback;
}

static bool has_json_key(const string& params, const string& key) {
    try {
        std::regex re("\\\"" + key + "\\\"\\s*:");
        return std::regex_search(params, re);
    } catch (...) {
        return false;
    }
}

static string quote_ident_cpp(const string& ident) {
    return string(quote_identifier(ident.c_str()));
}

static string qualified_relation_name(Oid relid) {
    char* relname = get_rel_name(relid);
    if (relname == NULL) {
        ereport(ERROR,
                (errcode(ERRCODE_UNDEFINED_TABLE),
                 errmsg("relation with oid %u does not exist", relid)));
    }

    Oid nspoid = get_rel_namespace(relid);
    char* nspname = get_namespace_name(nspoid);
    if (nspname == NULL) {
        ereport(ERROR,
                (errcode(ERRCODE_UNDEFINED_SCHEMA),
                 errmsg("schema for relation oid %u does not exist", relid)));
    }

    return quote_ident_cpp(nspname) + "." + quote_ident_cpp(relname);
}

static string sanitize_index_name(const string& name) {
    string out;
    out.reserve(name.size());
    for (char c : name) {
        if (std::isalnum((unsigned char)c) || c == '_' || c == '-' || c == '.') {
            out.push_back(c);
        } else {
            out.push_back('_');
        }
    }
    if (out.empty()) out = "faiss_index";
    return out;
}

static void ensure_dir_exists(const string& dir) {
    struct stat st;
    if (stat(dir.c_str(), &st) == 0) {
        if (!S_ISDIR(st.st_mode)) {
            ereport(ERROR,
                    (errmsg("Faiss index path exists but is not a directory: %s", dir.c_str())));
        }
        return;
    }

    if (mkdir(dir.c_str(), 0700) != 0) {
        int saved_errno = errno;
        const char* saved_error = strerror(saved_errno);
        ereport(ERROR,
                (errmsg("could not create Faiss index directory '%s': %s",
                        dir.c_str(), saved_error),
                 errhint("Create it manually and make it writable by the postgres user.")));
    }
}

static string build_index_path(const string& index_name, const string& params) {
    string dir = get_json_string_param(params, "indexDir", DEFAULT_INDEX_DIR);
    ensure_dir_exists(dir);
    if (!dir.empty() && dir.back() == '/') dir.pop_back();
    return dir + "/" + sanitize_index_name(index_name) + ".faiss";
}

// -----------------------------------------------------------------------------
// Conversão real[] -> float32
// -----------------------------------------------------------------------------

static int validate_real_array(ArrayType* array) {
    if (ARR_NDIM(array) != 1) {
        ereport(ERROR,
                (errcode(ERRCODE_ARRAY_SUBSCRIPT_ERROR),
                 errmsg("expected 1-D real[] array")));
    }
    if (ARR_ELEMTYPE(array) != FLOAT4OID) {
        ereport(ERROR,
                (errcode(ERRCODE_DATATYPE_MISMATCH),
                 errmsg("expected real[] (float4) array")));
    }
    if (ARR_HASNULL(array)) {
        ereport(ERROR,
                (errcode(ERRCODE_NULL_VALUE_NOT_ALLOWED),
                 errmsg("array elements must not be NULL")));
    }

    int nitems = ArrayGetNItems(ARR_NDIM(array), ARR_DIMS(array));
    if (nitems < 0) {
        ereport(ERROR, (errmsg("invalid real[] length")));
    }
    return nitems;
}

static void copy_real_array(ArrayType* array, int expected_dim, float* destination) {
    int nitems = validate_real_array(array);
    if (expected_dim >= 0 && nitems != expected_dim) {
        ereport(ERROR,
                (errmsg("embedding dimension (%d) differs from expected dimension (%d)",
                        nitems, expected_dim)));
    }

    if (nitems > 0) {
        std::memcpy(destination, ARR_DATA_PTR(array), (size_t)nitems * sizeof(float4));
    }
}

static void append_real_array(ArrayType* array, int expected_dim, vector<float>& destination) {
    int nitems = validate_real_array(array);
    if (expected_dim >= 0 && nitems != expected_dim) {
        ereport(ERROR,
                (errmsg("embedding dimension (%d) differs from expected dimension (%d)",
                        nitems, expected_dim)));
    }

    size_t old_size = destination.size();
    destination.resize(old_size + (size_t)nitems);
    if (nitems > 0) {
        std::memcpy(destination.data() + old_size,
                    ARR_DATA_PTR(array),
                    (size_t)nitems * sizeof(float4));
    }
}

static vector<float> pg_array_to_float_vector(ArrayType* array) {
    int nitems = validate_real_array(array);
    vector<float> out((size_t)nitems);
    if (nitems > 0) {
        std::memcpy(out.data(), ARR_DATA_PTR(array), (size_t)nitems * sizeof(float4));
    }
    return out;
}

static ArrayType* int_vector_to_pg_array(const vector<int32>& vals) {
    int nelems = (int) vals.size();
    Datum* elems = (Datum*) palloc((size_t)nelems * sizeof(Datum));
    for (int i = 0; i < nelems; i++) elems[i] = Int32GetDatum(vals[i]);

    int16 elmlen;
    bool elmbyval;
    char elmalign;
    get_typlenbyvalalign(INT4OID, &elmlen, &elmbyval, &elmalign);

    ArrayType* result = construct_array(
        elems, nelems, INT4OID, elmlen, elmbyval, elmalign);
    pfree(elems);
    return result;
}

static int64 get_int64_from_datum(Datum d, Oid type_oid) {
    if (type_oid == INT8OID) return DatumGetInt64(d);
    if (type_oid == INT4OID) return (int64) DatumGetInt32(d);

    ereport(ERROR,
            (errcode(ERRCODE_DATATYPE_MISMATCH),
             errmsg("id column must be int4 or int8 (got type oid %u)", type_oid)));
    return 0;
}

// -----------------------------------------------------------------------------
// Faiss: métricas, índices e parâmetros
// -----------------------------------------------------------------------------

static faiss::MetricType parse_metric_type(const string& metric_raw, bool* use_cosine) {
    string metric = to_lower_copy(metric_raw);
    *use_cosine = false;

    if (metric == "cosine" || metric == "cos" || metric == "angular") {
        *use_cosine = true;
        return faiss::METRIC_INNER_PRODUCT;
    }

    if (metric == "ip" || metric == "inner_product" ||
        metric == "inner-product" || metric == "dot" ||
        metric == "dot_product") {
        return faiss::METRIC_INNER_PRODUCT;
    }

    if (metric == "l2" || metric == "euclidean" || metric == "euclid") {
        return faiss::METRIC_L2;
    }

    ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
             errmsg("unsupported metric '%s' (use cosine, ip, or l2)",
                    metric_raw.c_str())));
    return faiss::METRIC_L2;
}

static string canonical_metric_name(const string& metric_raw) {
    string metric = to_lower_copy(metric_raw);
    if (metric == "cosine" || metric == "cos" || metric == "angular") return "cosine";
    if (metric == "ip" || metric == "inner_product" ||
        metric == "inner-product" || metric == "dot" ||
        metric == "dot_product") return "ip";
    if (metric == "l2" || metric == "euclidean" || metric == "euclid") return "l2";
    return metric;
}

static string canonical_index_type(const string& index_type_raw) {
    string t = to_lower_copy(index_type_raw);
    if (t.empty() || t == "flat" || t == "exact") return "flat";
    if (t == "hnsw" || starts_with(t, "hnsw")) return t;
    if (t == "ivf" || t == "ivfflat" || starts_with(t, "ivf")) return t;

    ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
             errmsg("unsupported index_type '%s' (use flat, hnsw32, or ivfflat)",
                    index_type_raw.c_str())));
    return "flat";
}

static int parse_hnsw_m(const string& index_type, const string& params) {
    int m = get_json_int_param(params, "M", 32);

    if (starts_with(index_type, "hnsw") && index_type.size() > 4) {
        try {
            int from_name = std::stoi(index_type.substr(4));
            if (from_name > 0) m = from_name;
        } catch (...) {}
    }

    if (m <= 0) {
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("HNSW parameter M must be positive")));
    }
    return m;
}

static int parse_ivf_nlist(const string& params, int64 ntotal) {
    int nlist = get_json_int_param(params, "nlist", 4096);
    if (nlist <= 0) {
        ereport(ERROR,
                (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                 errmsg("IVF parameter nlist must be positive")));
    }
    if (ntotal > 0 && (int64)nlist > ntotal) nlist = (int) ntotal;
    return nlist;
}

static faiss::Index* make_base_index(
    int d,
    faiss::MetricType metric,
    const string& index_type,
    int64 ntotal,
    const string& params) {

    if (index_type == "flat") {
        if (metric == faiss::METRIC_INNER_PRODUCT) return new faiss::IndexFlatIP(d);
        return new faiss::IndexFlatL2(d);
    }

    if (starts_with(index_type, "hnsw")) {
        int m = parse_hnsw_m(index_type, params);
        int ef_construction = get_json_int_param(params, "efConstruction", 200);
        int ef_search = get_json_int_param(params, "efSearch", 128);

        if (ef_construction <= 0 || ef_search <= 0) {
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                     errmsg("HNSW efConstruction and efSearch must be positive")));
        }

        faiss::IndexHNSWFlat* hnsw = new faiss::IndexHNSWFlat(d, m, metric);
        hnsw->hnsw.efConstruction = ef_construction;
        hnsw->hnsw.efSearch = ef_search;
        return hnsw;
    }

    if (starts_with(index_type, "ivf")) {
        int nlist = parse_ivf_nlist(params, ntotal);
        int nprobe = get_json_int_param(params, "nprobe", 64);
        if (nprobe <= 0) {
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                     errmsg("IVF parameter nprobe must be positive")));
        }

        faiss::Index* quantizer =
            (metric == faiss::METRIC_INNER_PRODUCT)
                ? (faiss::Index*) new faiss::IndexFlatIP(d)
                : (faiss::Index*) new faiss::IndexFlatL2(d);

        faiss::IndexIVFFlat* ivf =
            new faiss::IndexIVFFlat(quantizer, d, nlist, metric);
        ivf->own_fields = true;
        ivf->nprobe = (size_t) std::min(nprobe, nlist);
        return ivf;
    }

    ereport(ERROR,
            (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
             errmsg("unsupported index_type '%s'", index_type.c_str())));
    return NULL;
}

static faiss::Index* unwrap_id_map(faiss::Index* index) {
    if (faiss::IndexIDMap2* idmap2 = dynamic_cast<faiss::IndexIDMap2*>(index)) {
        return idmap2->index;
    }
    if (faiss::IndexIDMap* idmap = dynamic_cast<faiss::IndexIDMap*>(index)) {
        return idmap->index;
    }
    return index;
}

// Cria SearchParameters por chamada. Não altera o índice que está no cache.
static std::unique_ptr<faiss::SearchParameters> make_search_parameters(
    faiss::Index* index,
    const string& search_params) {

    faiss::Index* base = unwrap_id_map(index);

    if (faiss::IndexHNSW* hnsw = dynamic_cast<faiss::IndexHNSW*>(base)) {
        if (!has_json_key(search_params, "efSearch")) return nullptr;

        int ef_search = get_json_int_param(
            search_params, "efSearch", hnsw->hnsw.efSearch);
        if (ef_search <= 0) {
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                     errmsg("efSearch must be positive")));
        }

        std::unique_ptr<faiss::SearchParametersHNSW> p(
            new faiss::SearchParametersHNSW());
        p->efSearch = ef_search;
        return std::unique_ptr<faiss::SearchParameters>(p.release());
    }

    if (faiss::IndexIVF* ivf = dynamic_cast<faiss::IndexIVF*>(base)) {
        bool wants_nprobe = has_json_key(search_params, "nprobe");
        bool wants_max_codes = has_json_key(search_params, "maxCodes");
        if (!wants_nprobe && !wants_max_codes) return nullptr;

        int nprobe = get_json_int_param(
            search_params, "nprobe", (int) ivf->nprobe);
        int max_codes = get_json_int_param(search_params, "maxCodes", 0);

        if (nprobe <= 0) {
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                     errmsg("nprobe must be positive")));
        }
        if (max_codes < 0) {
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                     errmsg("maxCodes must be >= 0")));
        }

        std::unique_ptr<faiss::SearchParametersIVF> p(
            new faiss::SearchParametersIVF());
        p->nprobe = (size_t) std::min<int64>((int64)nprobe, (int64)ivf->nlist);
        p->max_codes = (size_t) max_codes;
        return std::unique_ptr<faiss::SearchParameters>(p.release());
    }

    return nullptr;
}

// -----------------------------------------------------------------------------
// Cache em memória por backend PostgreSQL
// -----------------------------------------------------------------------------

struct CachedFaissIndex {
    faiss::Index* index;
    int dim;
    string metric;
    string index_type;
    bool normalize_vectors;
    string index_path;
    TimestampTz updated_at;
};

struct LoadedIndexInfo {
    faiss::Index* index;
    int dim;
    string metric;
    string index_type;
    bool normalize_vectors;
    string index_path;
};

static std::unordered_map<string, CachedFaissIndex> faiss_cache;

static void erase_cache_entry(const string& index_name) {
    auto it = faiss_cache.find(index_name);
    if (it != faiss_cache.end()) {
        delete it->second.index;
        faiss_cache.erase(it);
    }
}

static void clear_all_cache_entries() {
    for (auto& kv : faiss_cache) delete kv.second.index;
    faiss_cache.clear();
}

static LoadedIndexInfo get_cached_index(const string& index_name) {
    if (SPI_connect() != SPI_OK_CONNECT) {
        ereport(ERROR, (errmsg("SPI_connect failed")));
    }

    const char* cmd =
        "SELECT dim, metric, index_type, normalize_vectors, index_path, updated_at, dirty "
        "FROM faiss_indexes WHERE name = $1";

    Oid argtypes[1] = { TEXTOID };
    Datum values[1] = { CStringGetTextDatum(index_name.c_str()) };
    const char nulls[1] = { ' ' };

    int ret = SPI_execute_with_args(
        cmd, 1, argtypes, values, nulls, true, 1);

    if (ret != SPI_OK_SELECT || SPI_processed != 1) {
        SPI_finish();
        ereport(ERROR,
                (errmsg("index '%s' not found in faiss_indexes", index_name.c_str())));
    }

    HeapTuple tup = SPI_tuptable->vals[0];
    TupleDesc tupdesc = SPI_tuptable->tupdesc;
    bool isnull = false;

    int32 dim = DatumGetInt32(SPI_getbinval(tup, tupdesc, 1, &isnull));
    if (isnull) {
        SPI_finish();
        ereport(ERROR, (errmsg("dim is NULL")));
    }

    Datum metricDatum = SPI_getbinval(tup, tupdesc, 2, &isnull);
    if (isnull) {
        SPI_finish();
        ereport(ERROR, (errmsg("metric is NULL")));
    }
    char* metricC = TextDatumGetCString(metricDatum);
    string metric(metricC);
    pfree(metricC);

    Datum indexTypeDatum = SPI_getbinval(tup, tupdesc, 3, &isnull);
    if (isnull) {
        SPI_finish();
        ereport(ERROR, (errmsg("index_type is NULL")));
    }
    char* indexTypeC = TextDatumGetCString(indexTypeDatum);
    string index_type(indexTypeC);
    pfree(indexTypeC);

    bool normalize_vectors =
        DatumGetBool(SPI_getbinval(tup, tupdesc, 4, &isnull));
    if (isnull) {
        SPI_finish();
        ereport(ERROR, (errmsg("normalize_vectors is NULL")));
    }

    Datum pathDatum = SPI_getbinval(tup, tupdesc, 5, &isnull);
    if (isnull) {
        SPI_finish();
        ereport(ERROR, (errmsg("index_path is NULL")));
    }
    char* pathC = TextDatumGetCString(pathDatum);
    string index_path(pathC);
    pfree(pathC);

    TimestampTz updated_at =
        DatumGetTimestampTz(SPI_getbinval(tup, tupdesc, 6, &isnull));
    if (isnull) {
        SPI_finish();
        ereport(ERROR, (errmsg("updated_at is NULL")));
    }

    bool dirty = DatumGetBool(SPI_getbinval(tup, tupdesc, 7, &isnull));
    if (!isnull && dirty) {
        SPI_finish();
        ereport(ERROR,
                (errmsg("index '%s' is dirty; run faiss_build_index again",
                        index_name.c_str())));
    }

    auto cached = faiss_cache.find(index_name);
    if (cached != faiss_cache.end() &&
        cached->second.updated_at == updated_at &&
        cached->second.index_path == index_path) {
        SPI_finish();
        return LoadedIndexInfo{
            cached->second.index,
            cached->second.dim,
            cached->second.metric,
            cached->second.index_type,
            cached->second.normalize_vectors,
            cached->second.index_path
        };
    }

    SPI_finish();

    faiss::Index* loaded = NULL;
    try {
        loaded = faiss::read_index(index_path.c_str());
    } catch (const std::exception& e) {
        ereport(ERROR,
                (errmsg("failed to read Faiss index from '%s': %s",
                        index_path.c_str(), e.what())));
    }

    if (loaded == NULL) {
        ereport(ERROR,
                (errmsg("failed to read Faiss index from '%s'", index_path.c_str())));
    }

    erase_cache_entry(index_name);
    faiss_cache[index_name] = CachedFaissIndex{
        loaded,
        dim,
        metric,
        index_type,
        normalize_vectors,
        index_path,
        updated_at
    };

    return LoadedIndexInfo{
        loaded, dim, metric, index_type, normalize_vectors, index_path
    };
}

// -----------------------------------------------------------------------------
// Funções antigas (compatibilidade)
// -----------------------------------------------------------------------------

extern "C" Datum faiss_knn_l2(PG_FUNCTION_ARGS) {
    ArrayType* queryArr = PG_GETARG_ARRAYTYPE_P(0);
    ArrayType* dataArr = PG_GETARG_ARRAYTYPE_P(1);
    int32 n = PG_GETARG_INT32(2);
    int32 d = PG_GETARG_INT32(3);
    int32 k = PG_GETARG_INT32(4);

    if (n <= 0 || d <= 0 || k <= 0) {
        ereport(ERROR, (errmsg("n, d and k must be positive")));
    }

    vector<float> query = pg_array_to_float_vector(queryArr);
    vector<float> data = pg_array_to_float_vector(dataArr);

    if ((int)query.size() != d) {
        ereport(ERROR,
                (errmsg("query vector length (%d) must equal d (%d)",
                        (int)query.size(), d)));
    }
    if ((int64)data.size() != (int64)n * (int64)d) {
        ereport(ERROR,
                (errmsg("data length (%ld) must equal n * d (%ld)",
                        (long)data.size(), (long)((int64)n * d))));
    }
    if (k > n) k = n;

    faiss::IndexFlatL2 index(d);
    index.add(n, data.data());

    vector<float> distances((size_t)k);
    vector<faiss::idx_t> labels((size_t)k);
    index.search(1, query.data(), k, distances.data(), labels.data());

    vector<int32> result_idx;
    result_idx.reserve((size_t)k);
    for (int i = 0; i < k; i++) {
        result_idx.push_back((int32)labels[i]);
    }

    PG_RETURN_ARRAYTYPE_P(int_vector_to_pg_array(result_idx));
}

extern "C" Datum faiss_knn_l2_table(PG_FUNCTION_ARGS) {
    ArrayType* queryArr = PG_GETARG_ARRAYTYPE_P(0);
    int32 k = PG_GETARG_INT32(1);
    if (k <= 0) ereport(ERROR, (errmsg("k must be positive")));

    vector<float> query = pg_array_to_float_vector(queryArr);
    int d = (int)query.size();
    if (d <= 0) ereport(ERROR, (errmsg("query vector must not be empty")));

    if (SPI_connect() != SPI_OK_CONNECT) {
        ereport(ERROR, (errmsg("SPI_connect failed")));
    }

    const char* cmd = "SELECT id, embedding FROM faiss_items ORDER BY id";
    int ret = SPI_execute(cmd, true, 0);
    if (ret != SPI_OK_SELECT) {
        SPI_finish();
        ereport(ERROR, (errmsg("SPI_execute failed with code %d", ret)));
    }

    uint64 nrows = SPI_processed;
    if (nrows == 0) {
        SPI_finish();
        vector<int32> empty;
        PG_RETURN_ARRAYTYPE_P(int_vector_to_pg_array(empty));
    }

    TupleDesc tupdesc = SPI_tuptable->tupdesc;
    SPITupleTable* tuptable = SPI_tuptable;
    Oid id_type = TupleDescAttr(tupdesc, 0)->atttypid;

    vector<int32> ids;
    vector<float> data;
    ids.reserve((size_t)nrows);
    data.reserve((size_t)nrows * (size_t)d);

    for (uint64 i = 0; i < nrows; i++) {
        HeapTuple tuple = tuptable->vals[i];
        bool isnull = false;

        Datum idDatum = SPI_getbinval(tuple, tupdesc, 1, &isnull);
        if (isnull) {
            SPI_finish();
            ereport(ERROR, (errmsg("id must not be NULL")));
        }
        int64 id64 = get_int64_from_datum(idDatum, id_type);
        if (id64 < std::numeric_limits<int32>::min() ||
            id64 > std::numeric_limits<int32>::max()) {
            SPI_finish();
            ereport(ERROR,
                    (errmsg("legacy faiss_knn_l2_table requires ids that fit in int4")));
        }

        Datum embDatum = SPI_getbinval(tuple, tupdesc, 2, &isnull);
        if (isnull) {
            SPI_finish();
            ereport(ERROR, (errmsg("embedding must not be NULL")));
        }

        ArrayType* embArray = DatumGetArrayTypeP(embDatum);
        append_real_array(embArray, d, data);
        ids.push_back((int32)id64);
    }

    SPI_finish();

    int n = (int)nrows;
    if (k > n) k = n;

    faiss::IndexFlatL2 index(d);
    index.add(n, data.data());

    vector<float> distances((size_t)k);
    vector<faiss::idx_t> labels((size_t)k);
    index.search(1, query.data(), k, distances.data(), labels.data());

    vector<int32> result_ids;
    result_ids.reserve((size_t)k);
    for (int i = 0; i < k; i++) {
        int idx = (int)labels[i];
        if (idx < 0 || idx >= n) {
            ereport(ERROR, (errmsg("Faiss returned invalid index %d", idx)));
        }
        result_ids.push_back(ids[(size_t)idx]);
    }

    PG_RETURN_ARRAYTYPE_P(int_vector_to_pg_array(result_ids));
}

// -----------------------------------------------------------------------------
// Helpers para construção em streaming
// -----------------------------------------------------------------------------

static Oid resolve_relation_name_via_spi(const string& relation_name) {
    const char* sql = "SELECT to_regclass($1)";
    Oid argtypes[1] = { TEXTOID };
    Datum values[1] = { CStringGetTextDatum(relation_name.c_str()) };
    const char nulls[1] = { ' ' };

    int ret = SPI_execute_with_args(
        sql, 1, argtypes, values, nulls, true, 1);
    if (ret != SPI_OK_SELECT || SPI_processed != 1) {
        ereport(ERROR,
                (errmsg("failed to resolve relation '%s'", relation_name.c_str())));
    }

    bool isnull = false;
    Datum d = SPI_getbinval(
        SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull);
    Oid oid = isnull ? InvalidOid : DatumGetObjectId(d);

    if (SPI_tuptable != NULL) SPI_freetuptable(SPI_tuptable);

    if (!OidIsValid(oid)) {
        ereport(ERROR,
                (errcode(ERRCODE_UNDEFINED_TABLE),
                 errmsg("trainingTable '%s' does not exist or is not visible in the current search_path",
                        relation_name.c_str())));
    }
    return oid;
}

static int64 read_table_count(const string& table_sql) {
    string sql = "SELECT count(*) FROM " + table_sql;
    int ret = SPI_execute(sql.c_str(), true, 0);

    if (ret != SPI_OK_SELECT || SPI_processed != 1) {
        ereport(ERROR,
                (errmsg("failed to count rows from %s", table_sql.c_str())));
    }

    bool isnull = false;
    int64 ntotal = DatumGetInt64(
        SPI_getbinval(SPI_tuptable->vals[0], SPI_tuptable->tupdesc, 1, &isnull));

    if (SPI_tuptable != NULL) SPI_freetuptable(SPI_tuptable);

    if (isnull || ntotal <= 0) {
        ereport(ERROR,
                (errmsg("%s is empty; nothing to index", table_sql.c_str())));
    }

    return ntotal;
}

// Reservoir sampling: mantém no máximo train_target vetores em RAM.
// O seed torna o treinamento reproduzível.
static vector<float> collect_ivf_training_sample(
    const string& table_sql,
    const string& id_column,
    const string& embedding_column,
    int64 ntotal,
    int train_target,
    int fetch_batch_size,
    uint64_t seed,
    bool normalize_vectors,
    int* d_out,
    int64* sampled_out) {

    string cmd =
        "SELECT " + quote_ident_cpp(embedding_column) +
        " FROM " + table_sql +
        " ORDER BY " + quote_ident_cpp(id_column);

    Portal portal = SPI_cursor_open_with_args(
        NULL, cmd.c_str(), 0, NULL, NULL, NULL, true, 0);
    if (portal == NULL) {
        ereport(ERROR, (errmsg("failed to open IVF training cursor")));
    }

    vector<float> sample;
    int d = -1;
    int64 sampled = 0;
    uint64_t seen = 0;
    std::mt19937_64 rng(seed);

    while (true) {
        SPI_cursor_fetch(portal, true, fetch_batch_size);
        uint64 batch_rows = SPI_processed;

        if (batch_rows == 0) {
            if (SPI_tuptable != NULL) SPI_freetuptable(SPI_tuptable);
            break;
        }

        SPITupleTable* tuptable = SPI_tuptable;
        TupleDesc tupdesc = tuptable->tupdesc;

        for (uint64 i = 0; i < batch_rows; i++) {
            HeapTuple tup = tuptable->vals[i];
            bool embnull = false;
            Datum embDatum = SPI_getbinval(tup, tupdesc, 1, &embnull);
            if (embnull) {
                SPI_freetuptable(tuptable);
                SPI_cursor_close(portal);
                ereport(ERROR, (errmsg("embedding must not be NULL")));
            }

            ArrayType* arr = DatumGetArrayTypeP(embDatum);
            int current_d = validate_real_array(arr);
            if (d < 0) {
                d = current_d;
                if (d <= 0) {
                    SPI_freetuptable(tuptable);
                    SPI_cursor_close(portal);
                    ereport(ERROR, (errmsg("embedding dimension must be > 0")));
                }
                sample.reserve((size_t)train_target * (size_t)d);
            } else if (current_d != d) {
                SPI_freetuptable(tuptable);
                SPI_cursor_close(portal);
                ereport(ERROR,
                        (errmsg("all embeddings must have same dimension (%d)", d)));
            }

            if (sampled < train_target) {
                size_t old = sample.size();
                sample.resize(old + (size_t)d);
                copy_real_array(arr, d, sample.data() + old);
                sampled++;
            } else {
                std::uniform_int_distribution<uint64_t> dist(0, seen);
                uint64_t j = dist(rng);
                if (j < (uint64_t)train_target) {
                    copy_real_array(arr, d, sample.data() + (size_t)j * (size_t)d);
                }
            }

            seen++;
        }

        SPI_freetuptable(tuptable);
    }

    SPI_cursor_close(portal);

    if ((int64)seen != ntotal) {
        ereport(ERROR,
                (errmsg("IVF training scan expected %ld rows but read %ld",
                        (long)ntotal, (long)seen)));
    }

    if (normalize_vectors && sampled > 0) {
        faiss::fvec_renorm_L2(d, (size_t)sampled, sample.data());
    }

    *d_out = d;
    *sampled_out = sampled;
    return sample;
}

static void ensure_dirty_trigger_for_table(Oid table_oid) {
    const char* sql = "SELECT faiss_ensure_dirty_trigger($1)";
    Oid argtypes[1] = { OIDOID };
    Datum values[1] = { ObjectIdGetDatum(table_oid) };
    const char nulls[1] = { ' ' };

    int ret = SPI_execute_with_args(
        sql, 1, argtypes, values, nulls, false, 0);

    if (ret != SPI_OK_SELECT) {
        ereport(ERROR,
                (errmsg("failed to install dirty-tracking trigger for source table (SPI code %d)", ret)));
    }

    if (SPI_tuptable != NULL) SPI_freetuptable(SPI_tuptable);
}

// -----------------------------------------------------------------------------
// Construção do índice em arquivo (streaming)
// -----------------------------------------------------------------------------

extern "C" Datum faiss_build_index(PG_FUNCTION_ARGS) {
    auto build_started = std::chrono::steady_clock::now();

    text* indexNameText = PG_GETARG_TEXT_PP(0);
    char* indexNameC = text_to_cstring(indexNameText);
    string indexName(indexNameC);
    pfree(indexNameC);

    bool legacy_mode = (PG_NARGS() == 1);

    Oid tableOid = InvalidOid;
    string tableSql;
    string idColumn = "id";
    string embeddingColumn = "embedding";
    string metricRaw = legacy_mode ? "l2" : text_arg_to_string(fcinfo, 4, "cosine");
    string indexTypeRaw = legacy_mode ? "flat" : text_arg_to_string(fcinfo, 5, "flat");
    bool normalize_vectors = legacy_mode ? false : PG_GETARG_BOOL(6);
    string params = legacy_mode ? "{}" : text_arg_to_string(fcinfo, 7, "{}");

    if (legacy_mode) {
        tableOid = RelnameGetRelid("faiss_items");
        if (!OidIsValid(tableOid)) {
            ereport(ERROR,
                    (errcode(ERRCODE_UNDEFINED_TABLE),
                     errmsg("legacy table faiss_items does not exist")));
        }
        tableSql = qualified_relation_name(tableOid);
    } else {
        tableOid = PG_GETARG_OID(1);
        Name idName = (Name)PG_GETARG_POINTER(2);
        Name embName = (Name)PG_GETARG_POINTER(3);
        tableSql = qualified_relation_name(tableOid);
        idColumn = string(NameStr(*idName));
        embeddingColumn = string(NameStr(*embName));
    }

    bool cosine_metric = false;
    faiss::MetricType metric = parse_metric_type(metricRaw, &cosine_metric);
    string metricName = canonical_metric_name(metricRaw);
    string indexType = canonical_index_type(indexTypeRaw);
    string indexPath = build_index_path(indexName, params);

    if (cosine_metric && !normalize_vectors) {
        ereport(WARNING,
                (errmsg("metric='cosine' requested with normalize_vectors=false; "
                        "results only match cosine if vectors are already normalized")));
    }

    int fetch_batch_size = get_json_int_param(params, "fetchBatchSize", 10000);
    if (fetch_batch_size <= 0) fetch_batch_size = 10000;

    if (SPI_connect() != SPI_OK_CONNECT) {
        ereport(ERROR, (errmsg("SPI_connect failed")));
    }

    // Impede INSERT/UPDATE/DELETE durante a fotografia usada para construir o índice.
    // DML usa ROW EXCLUSIVE, que conflita com SHARE.
    string lockSql = "LOCK TABLE " + tableSql + " IN SHARE MODE";
    int ret = SPI_execute(lockSql.c_str(), false, 0);
    if (ret != SPI_OK_UTILITY) {
        SPI_finish();
        ereport(ERROR,
                (errmsg("failed to lock source table %s", tableSql.c_str())));
    }

    int64 ntotal = read_table_count(tableSql);

    int d = -1;
    int64 training_rows = 0;
    Oid trainingTableOid = InvalidOid;
    string trainingTableSql;
    std::unique_ptr<faiss::Index> index;

    // IVF precisa ser treinado antes de receber add_with_ids().
    if (starts_with(indexType, "ivf")) {
        int nlist = parse_ivf_nlist(params, ntotal);

        // Por padrão a amostra de treinamento vem da própria base. Para seguir
        // benchmarks como Deep1B, pode-se informar uma tabela de learn separada:
        // {"trainingTable":"public.deep1b_learn", ...}
        string trainingTableParam = get_json_string_param(params, "trainingTable", "");
        string trainingIdColumn = get_json_string_param(params, "trainingIdColumn", idColumn);
        string trainingEmbeddingColumn = get_json_string_param(
            params, "trainingEmbeddingColumn", embeddingColumn);

        if (trainingTableParam.empty()) {
            trainingTableOid = tableOid;
            trainingTableSql = tableSql;
        } else {
            trainingTableOid = resolve_relation_name_via_spi(trainingTableParam);
            trainingTableSql = qualified_relation_name(trainingTableOid);
            if (trainingTableOid != tableOid) {
                string trainingLockSql =
                    "LOCK TABLE " + trainingTableSql + " IN SHARE MODE";
                int training_lock_ret = SPI_execute(trainingLockSql.c_str(), false, 0);
                if (training_lock_ret != SPI_OK_UTILITY) {
                    SPI_finish();
                    ereport(ERROR,
                            (errmsg("failed to lock IVF training table %s",
                                    trainingTableSql.c_str())));
                }
            }
        }

        int64 training_total = read_table_count(trainingTableSql);
        int default_train_size = std::max(100000, nlist * 40);
        if ((int64)default_train_size > training_total) {
            default_train_size = (int)training_total;
        }

        int train_size = get_json_int_param(params, "trainSize", default_train_size);
        if (train_size <= 0) {
            SPI_finish();
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                     errmsg("IVF trainSize must be positive")));
        }
        if ((int64)train_size > training_total) train_size = (int)training_total;
        if (train_size < nlist) {
            SPI_finish();
            ereport(ERROR,
                    (errcode(ERRCODE_INVALID_PARAMETER_VALUE),
                     errmsg("IVF trainSize (%d) must be >= nlist (%d)",
                            train_size, nlist)));
        }

        int seed_param = get_json_int_param(params, "trainSeed", 42);
        uint64_t train_seed = (uint64_t)(uint32_t)seed_param;

        vector<float> training;
        try {
            training = collect_ivf_training_sample(
                trainingTableSql,
                trainingIdColumn,
                trainingEmbeddingColumn,
                training_total,
                train_size,
                fetch_batch_size,
                train_seed,
                normalize_vectors,
                &d,
                &training_rows);

            index.reset(make_base_index(d, metric, indexType, ntotal, params));
            index->train((faiss::idx_t)training_rows, training.data());
        } catch (const std::exception& e) {
            index.reset();
            SPI_finish();
            ereport(ERROR,
                    (errmsg("Faiss IVF training failed: %s", e.what())));
        }

        // Libera explicitamente a amostra antes de percorrer a base para add().
        vector<float>().swap(training);
    }

    string cmd =
        "SELECT " + quote_ident_cpp(idColumn) + ", " +
        quote_ident_cpp(embeddingColumn) +
        " FROM " + tableSql +
        " ORDER BY " + quote_ident_cpp(idColumn);

    Portal portal = SPI_cursor_open_with_args(
        NULL, cmd.c_str(), 0, NULL, NULL, NULL, true, 0);
    if (portal == NULL) {
        index.reset();
        SPI_finish();
        ereport(ERROR, (errmsg("SPI_cursor_open_with_args failed")));
    }

    Oid id_type = InvalidOid;
    int64 rows_added = 0;

    vector<float> batch_data;
    vector<faiss::idx_t> batch_ids;

    while (true) {
        SPI_cursor_fetch(portal, true, fetch_batch_size);
        uint64 batch_rows = SPI_processed;

        if (batch_rows == 0) {
            if (SPI_tuptable != NULL) SPI_freetuptable(SPI_tuptable);
            break;
        }

        SPITupleTable* tuptable = SPI_tuptable;
        TupleDesc tupdesc = tuptable->tupdesc;

        if (id_type == InvalidOid) {
            id_type = TupleDescAttr(tupdesc, 0)->atttypid;
            if (id_type != INT4OID && id_type != INT8OID) {
                SPI_freetuptable(tuptable);
                SPI_cursor_close(portal);
                index.reset();
                SPI_finish();
                ereport(ERROR,
                        (errcode(ERRCODE_DATATYPE_MISMATCH),
                         errmsg("id column must be int4 or int8")));
            }
        }

        batch_data.clear();
        batch_ids.clear();
        batch_ids.reserve((size_t)batch_rows);
        if (d > 0) {
            batch_data.reserve((size_t)batch_rows * (size_t)d);
        }

        for (uint64 i = 0; i < batch_rows; i++) {
            HeapTuple tup = tuptable->vals[i];
            bool idnull = false;
            bool embnull = false;

            Datum idDatum = SPI_getbinval(tup, tupdesc, 1, &idnull);
            Datum embDatum = SPI_getbinval(tup, tupdesc, 2, &embnull);

            if (idnull || embnull) {
                SPI_freetuptable(tuptable);
                SPI_cursor_close(portal);
                index.reset();
                SPI_finish();
                ereport(ERROR, (errmsg("id/embedding must not be NULL")));
            }

            int64 id64 = get_int64_from_datum(idDatum, id_type);
            ArrayType* arr = DatumGetArrayTypeP(embDatum);
            int current_d = validate_real_array(arr);

            if (d < 0) {
                d = current_d;
                if (d <= 0) {
                    SPI_freetuptable(tuptable);
                    SPI_cursor_close(portal);
                    index.reset();
                    SPI_finish();
                    ereport(ERROR, (errmsg("embedding dimension must be > 0")));
                }

                batch_data.reserve((size_t)batch_rows * (size_t)d);

                // Flat/HNSW não precisam de treinamento. Criamos o índice quando
                // conhecemos a dimensão e o envolvemos em IndexIDMap (não IDMap2)
                // para reduzir overhead de memória.
                if (!index) {
                    try {
                        std::unique_ptr<faiss::Index> base(
                            make_base_index(d, metric, indexType, ntotal, params));
                        faiss::IndexIDMap* idmap = new faiss::IndexIDMap(base.release());
                        idmap->own_fields = true;
                        index.reset(idmap);
                    } catch (const std::exception& e) {
                        SPI_freetuptable(tuptable);
                        SPI_cursor_close(portal);
                        index.reset();
                        SPI_finish();
                        ereport(ERROR,
                                (errmsg("failed to create Faiss index: %s", e.what())));
                    }
                }
            } else if (current_d != d) {
                SPI_freetuptable(tuptable);
                SPI_cursor_close(portal);
                index.reset();
                SPI_finish();
                ereport(ERROR,
                        (errmsg("all embeddings must have same dimension (%d)", d)));
            }

            batch_ids.push_back((faiss::idx_t)id64);
            append_real_array(arr, d, batch_data);
        }

        if (normalize_vectors && !batch_ids.empty()) {
            faiss::fvec_renorm_L2(
                d, batch_ids.size(), batch_data.data());
        }

        try {
            index->add_with_ids(
                (faiss::idx_t)batch_ids.size(),
                batch_data.data(),
                batch_ids.data());
        } catch (const std::exception& e) {
            SPI_freetuptable(tuptable);
            SPI_cursor_close(portal);
            index.reset();
            SPI_finish();
            ereport(ERROR,
                    (errmsg("Faiss add_with_ids failed after %ld rows: %s",
                            (long)rows_added, e.what())));
        }

        rows_added += (int64)batch_ids.size();
        SPI_freetuptable(tuptable);
    }

    SPI_cursor_close(portal);

    if (!index || d <= 0) {
        index.reset();
        SPI_finish();
        ereport(ERROR, (errmsg("failed to create index")));
    }

    if (rows_added != ntotal) {
        index.reset();
        SPI_finish();
        ereport(ERROR,
                (errmsg("expected %ld rows but indexed %ld rows",
                        (long)ntotal, (long)rows_added)));
    }

    string tmpPath = indexPath + ".tmp";

    try {
        faiss::write_index(index.get(), tmpPath.c_str());
    } catch (const std::exception& e) {
        unlink(tmpPath.c_str());
        index.reset();
        SPI_finish();
        ereport(ERROR,
                (errmsg("failed to write Faiss index to '%s': %s",
                        tmpPath.c_str(), e.what())));
    }

    if (rename(tmpPath.c_str(), indexPath.c_str()) != 0) {
        int saved_errno = errno;
        const char* saved_error = strerror(saved_errno);
        unlink(tmpPath.c_str());
        index.reset();
        SPI_finish();
        ereport(ERROR,
                (errmsg("failed to rename '%s' to '%s': %s",
                        tmpPath.c_str(), indexPath.c_str(), saved_error)));
    }

    auto build_finished = std::chrono::steady_clock::now();
    double build_ms = std::chrono::duration<double, std::milli>(
        build_finished - build_started).count();

    const char* upsert =
        "INSERT INTO faiss_indexes("
        "name, dim, metric, index_type, normalize_vectors, ntotal, index_path, params, "
        "source_table_oid, source_table_name, id_column, embedding_column, "
        "training_source_table_oid, training_source_table_name, training_rows, build_ms, updated_at, dirty) "
        "VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,now(),false) "
        "ON CONFLICT (name) DO UPDATE SET "
        "dim=EXCLUDED.dim, metric=EXCLUDED.metric, index_type=EXCLUDED.index_type, "
        "normalize_vectors=EXCLUDED.normalize_vectors, ntotal=EXCLUDED.ntotal, "
        "index_path=EXCLUDED.index_path, params=EXCLUDED.params, "
        "source_table_oid=EXCLUDED.source_table_oid, source_table_name=EXCLUDED.source_table_name, "
        "id_column=EXCLUDED.id_column, embedding_column=EXCLUDED.embedding_column, "
        "training_source_table_oid=EXCLUDED.training_source_table_oid, "
        "training_source_table_name=EXCLUDED.training_source_table_name, "
        "training_rows=EXCLUDED.training_rows, build_ms=EXCLUDED.build_ms, "
        "updated_at=now(), dirty=false";

    Oid argtypes[16] = {
        TEXTOID, INT4OID, TEXTOID, TEXTOID, BOOLOID, INT8OID, TEXTOID, TEXTOID,
        OIDOID, TEXTOID, TEXTOID, TEXTOID, OIDOID, TEXTOID, INT8OID, FLOAT8OID
    };

    Datum values[16];
    char nulls[16] = {
        ' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' ',' '
    };

    values[0]  = CStringGetTextDatum(indexName.c_str());
    values[1]  = Int32GetDatum(d);
    values[2]  = CStringGetTextDatum(metricName.c_str());
    values[3]  = CStringGetTextDatum(indexType.c_str());
    values[4]  = BoolGetDatum(normalize_vectors);
    values[5]  = Int64GetDatum(ntotal);
    values[6]  = CStringGetTextDatum(indexPath.c_str());
    values[7]  = CStringGetTextDatum(params.c_str());
    values[8]  = ObjectIdGetDatum(tableOid);
    values[9]  = CStringGetTextDatum(tableSql.c_str());
    values[10] = CStringGetTextDatum(idColumn.c_str());
    values[11] = CStringGetTextDatum(embeddingColumn.c_str());
    if (OidIsValid(trainingTableOid)) {
        values[12] = ObjectIdGetDatum(trainingTableOid);
        values[13] = CStringGetTextDatum(trainingTableSql.c_str());
    } else {
        values[12] = (Datum)0;
        values[13] = (Datum)0;
        nulls[12] = 'n';
        nulls[13] = 'n';
    }
    values[14] = Int64GetDatum(training_rows);
    values[15] = Float8GetDatum(build_ms);

    ret = SPI_execute_with_args(
        upsert, 16, argtypes, values, nulls, false, 0);

    if (ret != SPI_OK_INSERT && ret != SPI_OK_UPDATE && ret != SPI_OK_INSERT_RETURNING) {
        index.reset();
        SPI_finish();
        ereport(ERROR,
                (errmsg("failed to upsert faiss_indexes (%d)", ret)));
    }
    if (SPI_tuptable != NULL) SPI_freetuptable(SPI_tuptable);

    ensure_dirty_trigger_for_table(tableOid);

    SPI_finish();

    // A cópia construída nesta chamada é liberada; a primeira busca carrega
    // o arquivo e passa a usar o cache por backend.
    index.reset();
    erase_cache_entry(indexName);

    PG_RETURN_VOID();
}

// -----------------------------------------------------------------------------
// Helpers de retorno de busca
// -----------------------------------------------------------------------------

static Tuplestorestate* begin_materialized_result(
    FunctionCallInfo fcinfo,
    TupleDesc* outdesc) {

    ReturnSetInfo* rsinfo = (ReturnSetInfo*)fcinfo->resultinfo;
    if (rsinfo == NULL || !IsA(rsinfo, ReturnSetInfo)) {
        ereport(ERROR,
                (errmsg("set-valued function called in a context that cannot accept a set")));
    }

    rsinfo->returnMode = SFRM_Materialize;

    if (get_call_result_type(fcinfo, NULL, outdesc) != TYPEFUNC_COMPOSITE) {
        ereport(ERROR, (errmsg("return type must be a composite type")));
    }

    if (rsinfo->econtext == NULL) {
        ereport(ERROR, (errmsg("no execution context available")));
    }

    MemoryContext oldcontext =
        MemoryContextSwitchTo(rsinfo->econtext->ecxt_per_query_memory);
    Tuplestorestate* tupstore =
        tuplestore_begin_heap(true, false, work_mem);
    MemoryContextSwitchTo(oldcontext);

    rsinfo->setResult = tupstore;
    rsinfo->setDesc = *outdesc;
    return tupstore;
}

// -----------------------------------------------------------------------------
// Busca de uma query
// -----------------------------------------------------------------------------

extern "C" Datum faiss_search(PG_FUNCTION_ARGS) {
    text* indexNameText = PG_GETARG_TEXT_PP(0);
    ArrayType* queryArr = PG_GETARG_ARRAYTYPE_P(1);
    int32 k = PG_GETARG_INT32(2);
    string searchParams = text_arg_to_string(fcinfo, 3, "{}");

    if (k <= 0) ereport(ERROR, (errmsg("k must be positive")));

    char* indexNameC = text_to_cstring(indexNameText);
    string indexName(indexNameC);
    pfree(indexNameC);

    vector<float> query = pg_array_to_float_vector(queryArr);
    int d = (int)query.size();
    if (d <= 0) ereport(ERROR, (errmsg("query must not be empty")));

    LoadedIndexInfo loaded = get_cached_index(indexName);
    faiss::Index* index = loaded.index;

    if (loaded.dim != d) {
        ereport(ERROR,
                (errmsg("query dim (%d) != index dim (%d)", d, loaded.dim)));
    }
    if (index->d != d) {
        ereport(ERROR,
                (errmsg("query dim (%d) != faiss index dim (%d)",
                        d, (int)index->d)));
    }

    if (loaded.normalize_vectors) {
        faiss::fvec_renorm_L2(d, 1, query.data());
    }

    if (index->ntotal <= 0) {
        ereport(ERROR, (errmsg("Faiss index is empty")));
    }
    if ((faiss::idx_t)k > index->ntotal) k = (int32)index->ntotal;

    std::unique_ptr<faiss::SearchParameters> local_params =
        make_search_parameters(index, searchParams);

    vector<float> distances((size_t)k);
    vector<faiss::idx_t> labels((size_t)k);

    try {
        index->search(
            1,
            query.data(),
            k,
            distances.data(),
            labels.data(),
            local_params.get());
    } catch (const std::exception& e) {
        ereport(ERROR, (errmsg("Faiss search failed: %s", e.what())));
    }

    TupleDesc outdesc;
    Tuplestorestate* tupstore = begin_materialized_result(fcinfo, &outdesc);

    ReturnSetInfo* rsinfo = (ReturnSetInfo*)fcinfo->resultinfo;
    MemoryContext oldcontext =
        MemoryContextSwitchTo(rsinfo->econtext->ecxt_per_query_memory);

    for (int i = 0; i < k; i++) {
        if (labels[(size_t)i] < 0) continue;

        Datum outvals[2];
        bool outnulls[2] = { false, false };
        outvals[0] = Int64GetDatum((int64)labels[(size_t)i]);
        outvals[1] = Float4GetDatum((float4)distances[(size_t)i]);
        tuplestore_putvalues(tupstore, outdesc, outvals, outnulls);
    }

    MemoryContextSwitchTo(oldcontext);
    PG_RETURN_NULL();
}

// -----------------------------------------------------------------------------
// Busca de várias queries em uma única chamada Faiss
//
// queries é um real[] achatado: [q0_dim0, ..., q0_dimD, q1_dim0, ...]
// query_no é 0-based para coincidir com os ids das queries do benchmark.
// -----------------------------------------------------------------------------

extern "C" Datum faiss_search_batch(PG_FUNCTION_ARGS) {
    text* indexNameText = PG_GETARG_TEXT_PP(0);
    ArrayType* queriesArr = PG_GETARG_ARRAYTYPE_P(1);
    int32 nq = PG_GETARG_INT32(2);
    int32 k = PG_GETARG_INT32(3);
    string searchParams = text_arg_to_string(fcinfo, 4, "{}");

    if (nq <= 0) ereport(ERROR, (errmsg("nq must be positive")));
    if (k <= 0) ereport(ERROR, (errmsg("k must be positive")));

    char* indexNameC = text_to_cstring(indexNameText);
    string indexName(indexNameC);
    pfree(indexNameC);

    vector<float> queries = pg_array_to_float_vector(queriesArr);
    if (queries.empty()) ereport(ERROR, (errmsg("queries must not be empty")));

    if (queries.size() % (size_t)nq != 0) {
        ereport(ERROR,
                (errmsg("queries length (%ld) is not divisible by nq (%d)",
                        (long)queries.size(), nq)));
    }

    int d = (int)(queries.size() / (size_t)nq);
    if (d <= 0) ereport(ERROR, (errmsg("query dimension must be positive")));

    LoadedIndexInfo loaded = get_cached_index(indexName);
    faiss::Index* index = loaded.index;

    if (loaded.dim != d || index->d != d) {
        ereport(ERROR,
                (errmsg("batch query dim (%d) != index dim (%d)", d, loaded.dim)));
    }

    if (loaded.normalize_vectors) {
        faiss::fvec_renorm_L2(d, (size_t)nq, queries.data());
    }

    if (index->ntotal <= 0) {
        ereport(ERROR, (errmsg("Faiss index is empty")));
    }
    if ((faiss::idx_t)k > index->ntotal) k = (int32)index->ntotal;

    size_t result_count = (size_t)nq * (size_t)k;
    if (k > 0 && result_count / (size_t)k != (size_t)nq) {
        ereport(ERROR, (errmsg("batch result size overflow")));
    }

    std::unique_ptr<faiss::SearchParameters> local_params =
        make_search_parameters(index, searchParams);

    vector<float> distances(result_count);
    vector<faiss::idx_t> labels(result_count);

    try {
        index->search(
            (faiss::idx_t)nq,
            queries.data(),
            k,
            distances.data(),
            labels.data(),
            local_params.get());
    } catch (const std::exception& e) {
        ereport(ERROR, (errmsg("Faiss batch search failed: %s", e.what())));
    }

    TupleDesc outdesc;
    Tuplestorestate* tupstore = begin_materialized_result(fcinfo, &outdesc);

    ReturnSetInfo* rsinfo = (ReturnSetInfo*)fcinfo->resultinfo;
    MemoryContext oldcontext =
        MemoryContextSwitchTo(rsinfo->econtext->ecxt_per_query_memory);

    for (int32 q = 0; q < nq; q++) {
        size_t base = (size_t)q * (size_t)k;
        for (int32 rank = 0; rank < k; rank++) {
            size_t pos = base + (size_t)rank;
            if (labels[pos] < 0) continue;

            Datum outvals[3];
            bool outnulls[3] = { false, false, false };
            outvals[0] = Int32GetDatum(q);
            outvals[1] = Int64GetDatum((int64)labels[pos]);
            outvals[2] = Float4GetDatum((float4)distances[pos]);
            tuplestore_putvalues(tupstore, outdesc, outvals, outnulls);
        }
    }

    MemoryContextSwitchTo(oldcontext);
    PG_RETURN_NULL();
}

extern "C" Datum faiss_clear_cache(PG_FUNCTION_ARGS) {
    text* indexNameText = PG_GETARG_TEXT_PP(0);
    char* indexNameC = text_to_cstring(indexNameText);
    string indexName(indexNameC);
    pfree(indexNameC);

    erase_cache_entry(indexName);
    PG_RETURN_VOID();
}

extern "C" Datum faiss_clear_all_cache(PG_FUNCTION_ARGS) {
    clear_all_cache_entries();
    PG_RETURN_VOID();
}
