# ============================================================
# DGA DOMAIN DETECTOR - FLASK MULTI-MODEL BACKEND
# UPDATED FOR 10 LEXICAL FEATURES
# UI / TEMPLATE STRUCTURE UNCHANGED
# ============================================================

from flask import Flask, render_template, request

import os
import re
import json
import math
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


# ============================================================
# APP
# ============================================================

app = Flask(__name__)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

FEATURE_DIR = os.path.join(
    BASE_DIR,
    "features"
)


# ============================================================
# DEPLOYMENT CONFIG
# ============================================================

CONFIG_PATH = os.path.join(
    FEATURE_DIR,
    "deployment_config.json"
)

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:
    config = json.load(file)


MAX_LEN = int(
    config.get(
        "max_len",
        31
    )
)


# ============================================================
# THRESHOLDS
# ============================================================

# Weighted Hybrid CNN threshold selected from validation
HYBRID_THRESHOLD = float(
    config.get(
        "weighted_hybrid_threshold",
        0.20
    )
)

# Other individual models
DEFAULT_MODEL_THRESHOLD = 0.50


# ============================================================
# FINAL VALIDATED ENSEMBLE
# ============================================================

ensemble_config = config.get(
    "ensemble",
    {}
)

FINAL_LR_WEIGHT = float(
    ensemble_config.get(
        "lr_weight",
        0.50
    )
)

FINAL_HYBRID_WEIGHT = float(
    ensemble_config.get(
        "cnn_weight",
        0.50
    )
)

ENSEMBLE_THRESHOLD = float(
    ensemble_config.get(
        "threshold",
        0.50
    )
)


# ============================================================
# FEATURE NAMES — UPDATED 10 FEATURES
# ============================================================

DEFAULT_FEATURE_NAMES = [
    "domain_length",
    "digit_count",
    "digit_ratio",
    "vowel_ratio",
    "consonant_ratio",
    "max_consecutive_consonant",
    "max_consecutive_digit",
    "entropy",
    "special_character_count",
    "special_character_ratio"
]

FEATURE_NAMES = config.get(
    "lexical_features",
    DEFAULT_FEATURE_NAMES
)

# Safety check
if len(FEATURE_NAMES) != 10:
    raise RuntimeError(
        f"Expected 10 lexical features, "
        f"but deployment config contains {len(FEATURE_NAMES)}."
    )


FEATURE_LABELS = {
    "domain_length": "Domain Length",
    "digit_count": "Digit Count",
    "digit_ratio": "Digit Ratio",
    "vowel_ratio": "Vowel Ratio",
    "consonant_ratio": "Consonant Ratio",
    "max_consecutive_consonant": "Longest Consonant Run",
    "max_consecutive_digit": "Longest Digit Run",
    "entropy": "Entropy",
    "special_character_count": "Special Character Count",
    "special_character_ratio": "Special Character Ratio"
}


# ============================================================
# MODEL INFO
# IMPORTANT:
# Model names remain unchanged so the current UI keeps working.
# ============================================================

MODEL_INFO = {
    "Logistic Regression + TF-IDF": {
        "short": "LR",
        "input": "Character TF-IDF",
        "type": "Logistic Regression"
    },

    "Random Forest + Lexical": {
        "short": "RF",
        "input": "10 Lexical Features",
        "type": "Random Forest"
    },

    "XGBoost + Lexical": {
        "short": "XG",
        "input": "10 Lexical Features",
        "type": "XGBoost"
    },

    "Lexical Neural Network": {
        "short": "NN",
        "input": "10 Lexical Features",
        "type": "Lexical Neural Network"
    },

    "Character CNN": {
        "short": "CNN",
        "input": "Character Sequence",
        "type": "Character CNN"
    },

    "Hybrid CNN + Lexical": {
        "short": "HY",
        "input": "Character + Lexical",
        "type": "Weighted Hybrid CNN"
    }
}


VOWELS = set(
    "aeiou"
)


# ============================================================
# LOAD CHARACTER TOKENIZER
# ============================================================

TOKENIZER_PATH = os.path.join(
    FEATURE_DIR,
    "character_tokenizer.json"
)

with open(
    TOKENIZER_PATH,
    "r",
    encoding="utf-8"
) as file:
    tokenizer = tokenizer_from_json(
        file.read()
    )


# ============================================================
# LOAD TF-IDF VECTORIZER
# ============================================================

TFIDF_PATH = os.path.join(
    FEATURE_DIR,
    "char_tfidf_vectorizer.pkl"
)

tfidf_vectorizer = joblib.load(
    TFIDF_PATH
)


# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 60)
print("Loading DGA detection models...")
print("=" * 60)


logistic_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "logistic_tfidf.pkl"
    )
)


random_forest_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "random_forest_lexical.pkl"
    )
)


xgboost_model = joblib.load(
    os.path.join(
        MODEL_DIR,
        "xgboost_lexical.pkl"
    )
)


lexical_model = tf.keras.models.load_model(
    os.path.join(
        MODEL_DIR,
        "lexical_only_best.keras"
    ),
    compile=False
)


character_model = tf.keras.models.load_model(
    os.path.join(
        MODEL_DIR,
        "character_cnn_only_best.keras"
    ),
    compile=False
)


hybrid_model = tf.keras.models.load_model(
    os.path.join(
        MODEL_DIR,
        "hybrid_cnn_weighted_best.keras"
    ),
    compile=False
)


print("All 6 models loaded successfully.")
print("Lexical feature count:", len(FEATURE_NAMES))
print("MAX_LEN:", MAX_LEN)
print("Hybrid threshold:", HYBRID_THRESHOLD)
print(
    "Final ensemble weights:",
    FINAL_LR_WEIGHT,
    FINAL_HYBRID_WEIGHT
)
print("Ensemble threshold:", ENSEMBLE_THRESHOLD)
print("=" * 60)


# ============================================================
# DOMAIN NORMALIZATION
# ============================================================

def normalize_domain(value):

    domain = str(
        value or ""
    ).strip().lower()

    domain = re.sub(
        r"^[a-z][a-z0-9+.-]*://",
        "",
        domain
    )

    domain = domain.split("/")[0]
    domain = domain.split("?")[0]
    domain = domain.split("#")[0]

    if "@" in domain:
        domain = domain.rsplit(
            "@",
            1
        )[-1]

    if domain.count(":") == 1:
        domain = domain.split(":")[0]

    return domain.strip(".")


# ============================================================
# ENTROPY
# ============================================================

def calculate_entropy(domain):

    if len(domain) == 0:
        return 0.0

    counts = Counter(
        domain
    )

    entropy_value = 0.0

    for count in counts.values():

        probability = (
            count
            /
            len(domain)
        )

        entropy_value -= (
            probability
            *
            math.log2(probability)
        )

    return entropy_value


# ============================================================
# MAX CONSECUTIVE CHARACTER TYPE
# ============================================================

def max_consecutive(
    text,
    condition
):

    maximum = 0
    current = 0

    for character in text:

        if condition(character):

            current += 1

            maximum = max(
                maximum,
                current
            )

        else:

            current = 0

    return maximum


# ============================================================
# EXACT 10 FEATURE EXTRACTION
# MUST MATCH THE UPDATED COLAB NOTEBOOK
# ============================================================

def extract_features(domain):

    domain = str(
        domain
    ).lower()

    length = len(
        domain
    )

    digit_count = sum(
        character.isdigit()
        for character in domain
    )

    vowel_count = sum(
        character in VOWELS
        for character in domain
    )

    consonant_count = sum(
        character.isalpha()
        and character not in VOWELS
        for character in domain
    )

    special_character_count = sum(
        not character.isalnum()
        for character in domain
    )


    features = [

        # 1 — Domain Length
        length,

        # 2 — Digit Count
        digit_count,

        # 3 — Digit Ratio
        digit_count
        /
        max(length, 1),

        # 4 — Vowel Ratio
        vowel_count
        /
        max(length, 1),

        # 5 — Consonant Ratio
        consonant_count
        /
        max(length, 1),

        # 6 — Maximum Consecutive Consonants
        max_consecutive(
            domain,
            lambda character:
            character.isalpha()
            and character not in VOWELS
        ),

        # 7 — Maximum Consecutive Digits
        max_consecutive(
            domain,
            lambda character:
            character.isdigit()
        ),

        # 8 — Shannon Entropy
        calculate_entropy(
            domain
        ),

        # 9 — Special Character Count
        special_character_count,

        # 10 — Special Character Ratio
        special_character_count
        /
        max(length, 1)
    ]

    return features


# ============================================================
# FEATURE DISPLAY CONFIGURATION
# ============================================================

PERCENTAGE_FEATURES = {
    "digit_ratio",
    "vowel_ratio",
    "consonant_ratio",
    "special_character_ratio"
}


INTEGER_FEATURES = {
    "domain_length",
    "digit_count",
    "max_consecutive_consonant",
    "max_consecutive_digit",
    "special_character_count"
}


BINARY_FEATURES = set()


# ============================================================
# FORMAT FEATURE VALUES
# ============================================================

def format_feature_value(
    name,
    value
):

    if name in PERCENTAGE_FEATURES:

        return (
            f"{float(value) * 100:.2f}%"
        )

    if name in INTEGER_FEATURES:

        return str(
            int(value)
        )

    if name in BINARY_FEATURES:

        return (
            "Yes"
            if int(value)
            else "No"
        )

    return (
        f"{float(value):.4f}"
    )


# ============================================================
# FEATURE ROWS FOR UI
# ============================================================

def build_feature_rows(
    feature_values
):

    rows = []

    for name, value in zip(
        FEATURE_NAMES,
        feature_values
    ):

        rows.append({

            "name":
            name,

            "label":
            FEATURE_LABELS.get(
                name,
                name
            ),

            "value":
            format_feature_value(
                name,
                value
            ),

            "raw":
            float(value)

        })

    return rows


# ============================================================
# DOMAIN SIGNALS
# UPDATED FOR 10 FEATURES
# ============================================================

def build_domain_signals(
    feature_values
):

    feature_map = dict(
        zip(
            FEATURE_NAMES,
            feature_values
        )
    )

    signals = [

        (
            "Domain length: "
            f"{int(feature_map['domain_length'])} characters"
        ),

        (
            "Shannon entropy: "
            f"{feature_map['entropy']:.3f}"
        ),

        (
            "Digit ratio: "
            f"{feature_map['digit_ratio'] * 100:.1f}%"
        ),

        (
            "Vowel ratio: "
            f"{feature_map['vowel_ratio'] * 100:.1f}%"
        ),

        (
            "Consonant ratio: "
            f"{feature_map['consonant_ratio'] * 100:.1f}%"
        ),

        (
            "Longest consonant run: "
            f"{int(feature_map['max_consecutive_consonant'])}"
        ),

        (
            "Longest digit run: "
            f"{int(feature_map['max_consecutive_digit'])}"
        ),

        (
            "Special character ratio: "
            f"{feature_map['special_character_ratio'] * 100:.1f}%"
        )

    ]

    return signals


# ============================================================
# DOMAIN PREDICTION
# ============================================================

def predict_domain(domain):

    domain = normalize_domain(
        domain
    )


    if not domain:

        raise ValueError(
            "Please enter a valid domain name."
        )


    # ========================================================
    # TF-IDF INPUT
    # ========================================================

    tfidf_input = (
        tfidf_vectorizer
        .transform(
            [domain]
        )
    )


    # ========================================================
    # LEXICAL INPUT — UPDATED 10 FEATURES
    # ========================================================

    feature_values = extract_features(
        domain
    )


    lexical_dataframe = pd.DataFrame(
        [
            feature_values
        ],
        columns=FEATURE_NAMES
    )


    lexical_array = (
        lexical_dataframe
        .values
        .astype(
            np.float32
        )
    )


    # ========================================================
    # CHARACTER INPUT
    # ========================================================

    sequence = (
        tokenizer
        .texts_to_sequences(
            [domain]
        )
    )


    character_input = pad_sequences(

        sequence,

        maxlen=MAX_LEN,

        padding="post",

        truncating="post"

    )


    # ========================================================
    # ACTUAL MODEL PROBABILITIES
    # ========================================================

    predictions = {

        "Logistic Regression + TF-IDF":

        float(
            logistic_model
            .predict_proba(
                tfidf_input
            )[0][1]
        ),


        "Random Forest + Lexical":

        float(
            random_forest_model
            .predict_proba(
                lexical_dataframe
            )[0][1]
        ),


        "XGBoost + Lexical":

        float(
            xgboost_model
            .predict_proba(
                lexical_dataframe
            )[0][1]
        ),


        "Lexical Neural Network":

        float(
            lexical_model
            .predict(
                lexical_array,
                verbose=0
            )[0][0]
        ),


        "Character CNN":

        float(
            character_model
            .predict(
                character_input,
                verbose=0
            )[0][0]
        ),


        "Hybrid CNN + Lexical":

        float(
            hybrid_model
            .predict(
                [
                    character_input,
                    lexical_array
                ],
                verbose=0
            )[0][0]
        )
    }


    # ========================================================
    # FINAL LR + WEIGHTED HYBRID ENSEMBLE
    # VALIDATION SELECTED: 0.50 / 0.50
    # ========================================================

    logistic_probability = predictions[
        "Logistic Regression + TF-IDF"
    ]


    hybrid_probability = predictions[
        "Hybrid CNN + Lexical"
    ]


    final_probability = (

        FINAL_LR_WEIGHT
        *
        logistic_probability

        +

        FINAL_HYBRID_WEIGHT
        *
        hybrid_probability

    )


    # ========================================================
    # FINAL CLASSIFICATION
    # ========================================================

    if final_probability >= ENSEMBLE_THRESHOLD:

        final_prediction = "DGA"

    else:

        final_prediction = "BENIGN"


    threat_score = round(
        final_probability
        *
        100,
        2
    )


    # ========================================================
    # RISK LEVEL
    # ========================================================

    if threat_score >= 70:

        risk = "HIGH"

    elif threat_score >= 30:

        risk = "MEDIUM"

    else:

        risk = "LOW"


    # ========================================================
    # INDIVIDUAL MODEL RESULTS
    # ========================================================

    model_results = []


    for name, probability in predictions.items():

        # Weighted Hybrid uses validation-selected 0.20 threshold.
        # Other individual models retain 0.50 threshold.
        if name == "Hybrid CNN + Lexical":

            model_threshold = HYBRID_THRESHOLD

        else:

            model_threshold = DEFAULT_MODEL_THRESHOLD


        model_prediction = (

            "DGA"

            if probability >= model_threshold

            else "BENIGN"

        )


        model_metadata = MODEL_INFO[
            name
        ]


        model_results.append({

            "model":
            name,

            "short":
            model_metadata[
                "short"
            ],

            "input":
            model_metadata[
                "input"
            ],

            "type":
            model_metadata[
                "type"
            ],

            "prediction":
            model_prediction,

            "probability":
            round(
                probability * 100,
                2
            ),

            "threshold":
            round(
                model_threshold * 100,
                2
            )

        })


    # ========================================================
    # MODEL CONSENSUS
    # ========================================================

    agreement_count = sum(

        model[
            "prediction"
        ]
        ==
        final_prediction

        for model in model_results

    )


    dga_votes = sum(

        model[
            "prediction"
        ]
        ==
        "DGA"

        for model in model_results

    )


    benign_votes = (
        len(model_results)
        -
        dga_votes
    )


    # ========================================================
    # RETURN
    # ========================================================

    return {

        "domain":
        domain,

        "prediction":
        final_prediction,

        "risk":
        risk,

        "threat_score":
        threat_score,

        "models":
        model_results,

        "agreement_count":
        agreement_count,

        "model_count":
        len(model_results),

        "dga_votes":
        dga_votes,

        "benign_votes":
        benign_votes,

        "threshold":
        round(
            ENSEMBLE_THRESHOLD
            *
            100,
            2
        ),

        "final_engine":
        (
            f"Logistic Regression "
            f"{FINAL_LR_WEIGHT * 100:.0f}% + "
            f"Weighted Hybrid CNN "
            f"{FINAL_HYBRID_WEIGHT * 100:.0f}%"
        ),

        "features":
        build_feature_rows(
            feature_values
        ),

        "signals":
        build_domain_signals(
            feature_values
        )

    }


# ============================================================
# FLASK ROUTE
# ============================================================

@app.route(
    "/",
    methods=[
        "GET",
        "POST"
    ]
)
def home():

    context = {

        "domain_name":
        "",

        "prediction":
        "",

        "risk":
        "",

        "threat_score":
        None,

        "models":
        [],

        "feature_info":
        [],

        "signals":
        [],

        "agreement_count":
        0,

        "model_count":
        6,

        "dga_votes":
        0,

        "benign_votes":
        0,

        "threshold":
        ENSEMBLE_THRESHOLD * 100,

        "final_engine":
        "",

        "error":
        ""

    }


    if request.method == "POST":

        submitted_domain = request.form.get(
            "domain",
            ""
        )


        try:

            result = predict_domain(
                submitted_domain
            )


            context[
                "domain_name"
            ] = result[
                "domain"
            ]


            context[
                "prediction"
            ] = result[
                "prediction"
            ]


            context[
                "risk"
            ] = result[
                "risk"
            ]


            context[
                "threat_score"
            ] = result[
                "threat_score"
            ]


            context[
                "models"
            ] = result[
                "models"
            ]


            context[
                "feature_info"
            ] = result[
                "features"
            ]


            context[
                "signals"
            ] = result[
                "signals"
            ]


            context[
                "agreement_count"
            ] = result[
                "agreement_count"
            ]


            context[
                "model_count"
            ] = result[
                "model_count"
            ]


            context[
                "dga_votes"
            ] = result[
                "dga_votes"
            ]


            context[
                "benign_votes"
            ] = result[
                "benign_votes"
            ]


            context[
                "threshold"
            ] = result[
                "threshold"
            ]


            context[
                "final_engine"
            ] = result[
                "final_engine"
            ]


        except Exception as error:

            app.logger.exception(
                "Domain analysis failed"
            )


            context[
                "error"
            ] = str(
                error
            )


            context[
                "domain_name"
            ] = normalize_domain(
                submitted_domain
            )


    return render_template(
        "index.html",
        **context
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print(
        "Starting DGA Domain Detector..."
    )

    print(
        "Open: http://127.0.0.1:8080"
    )

    app.run(
        host="127.0.0.1",
        port=8080,
        debug=True
    )