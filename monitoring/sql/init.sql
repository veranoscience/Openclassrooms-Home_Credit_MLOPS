BEGIN;

DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS errors CASCADE;

-- =========================
-- SUCCÈS /predict
-- =========================
CREATE TABLE predictions (
    request_id      TEXT PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL,          
    client_id       TEXT,                         
    model_name      TEXT NOT NULL,
    model_version   TEXT,

    threshold       DOUBLE PRECISION NOT NULL,
    fn_cost         DOUBLE PRECISION,
    fp_cost         DOUBLE PRECISION,

    latency_ms      DOUBLE PRECISION NOT NULL,
    status_code     INTEGER NOT NULL DEFAULT 200, 

    proba           DOUBLE PRECISION NOT NULL,     
    prediction      SMALLINT NOT NULL CHECK (prediction IN (0,1)),
    decision        TEXT NOT NULL CHECK (decision IN ('Accepté','Refusé')),

    missing_rate    DOUBLE PRECISION,             
    missing_count   INTEGER,                      
    n_features_sent INTEGER,                   
    payload_hash    TEXT,

    top_features    JSONB, 
    input_json      JSONB,  
    output_json     JSONB 
);

CREATE INDEX idx_predictions_ts     ON predictions(ts);
CREATE INDEX idx_predictions_model  ON predictions(model_name, model_version);
CREATE INDEX idx_predictions_status ON predictions(status_code);

-- =========================
-- ERREURS /predict
-- =========================
CREATE TABLE errors (
    request_id      TEXT PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL,       
    client_id       TEXT,
    model_name      TEXT,
    model_version   TEXT,
    latency_ms      DOUBLE PRECISION,
    status_code     INTEGER NOT NULL,         
    payload_hash    TEXT,
    error_json      JSONB,
    input_json      JSONB
);

CREATE INDEX idx_errors_ts     ON errors(ts);
CREATE INDEX idx_errors_status ON errors(status_code);

COMMIT;
