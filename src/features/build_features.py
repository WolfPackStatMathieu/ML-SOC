from src.utils.url_utils import (
    count_dot,
    no_of_dir,
    no_of_embed,
    shortening_service,
    count_http,
    count_per,
    count_ques,
    count_hyphen,
    count_equal,
    hostname_length,
    suspicious_words,
    digit_count,
    letter_count,
    url_length,
    number_of_parameters,
    number_of_fragments,
    is_encoded,
    count_special_characters,
    unusual_character_ratio,
)
from src.utils.content_utils import (
    apply_to_content,
    count_dot as count_dot_content,
    no_of_dir as count_dir_content,
    no_of_embed as count_embed_domain_content,
    count_per as count_per_content,
    count_ques as count_ques_content,
    count_hyphen as count_hyphen_content,
    count_equal as count_equal_content,
    url_length as content_length,
    suspicious_words as sus_content,
    digit_count as count_digits_content,
    letter_count as count_letters_content,
    count_special_characters as special_count_content,
    is_encoded as is_encoded_content,
)
from sklearn.preprocessing import LabelEncoder


def build_features(data):
    """
    Preprocess and extract features from the raw data.

    Parameters:
    data (pd.DataFrame): The raw data.

    Returns:
    pd.DataFrame: The features.
    pd.Series: The target variable.
    """
    feature_names = [
        "Unnamed: 0",
        "Method",
        "User-Agent",
        "Pragma",
        "Cache-Control",
        "Accept",
        "Accept-encoding",
        "Accept-charset",
        "language",
        "host",
        "cookie",
        "content-type",
        "connection",
        "lenght",
        "content",
        "classification",
        "URL",
    ]

    X = data[feature_names]
    X = X.rename(columns={"Unnamed: 0": "Class", "lenght": "content_length"})

    selected_features = [
        "Class",
        "Method",
        "host",
        "cookie",
        "Accept",
        "content_length",
        "content",
        "classification",
        "URL",
    ]

    X = X[selected_features]
    y = X["Class"]

    X["content_length"] = (
        X["content_length"].astype(str).str.extract(r"(\d+)").fillna(0).astype(int)
    )

    url_feature_functions = {
        "count_dot_url": count_dot,
        "count_dir_url": no_of_dir,
        "count_embed_domain_url": no_of_embed,
        "shortening_service_url": shortening_service,
        "count_http_url": count_http,
        "count%_url": count_per,
        "count?_url": count_ques,
        "count-_url": count_hyphen,
        "count=_url": count_equal,
        "url_length": url_length,
        "hostname_length_url": hostname_length,
        "sus_url": suspicious_words,
        "count_digits_url": digit_count,
        "count_letters_url": letter_count,
        "number_of_parameters_url": number_of_parameters,
        "number_of_fragments_url": number_of_fragments,
        "is_encoded_url": is_encoded,
        "special_count_url": count_special_characters,
        "unusual_character_ratio_url": unusual_character_ratio,
    }

    for feature, func in url_feature_functions.items():
        X[feature] = X["URL"].apply(func)

    # Convert 'content' column to string to avoid issues with float
    X["content"] = X["content"].astype(str)

    content_feature_functions = {
        "count_dot_content": count_dot_content,
        "count_dir_content": count_dir_content,
        "count_embed_domain_content": count_embed_domain_content,
        "count%_content": count_per_content,
        "count?_content": count_ques_content,
        "count-_content": count_hyphen_content,
        "count=_content": count_equal_content,
        "sus_content": sus_content,
        "count_digits_content": count_digits_content,
        "count_letters_content": count_letters_content,
        "content_length": content_length,
        "is_encoded_content": is_encoded_content,
        "special_count_content": special_count_content,
    }

    for feature, func in content_feature_functions.items():
        X[feature] = X["content"].apply(lambda x: apply_to_content(x, func))

    # Encode categorical features
    categorical_features = ["Method", "host", "cookie", "Accept", "content",
                            "URL", "classification"]
    le = LabelEncoder()
    for feature in categorical_features:
        X[feature] = le.fit_transform(X[feature].astype(str))

    # Encode target variable
    y = le.fit_transform(y.astype(str))

    # Ensure categorical features are treated as strings for imputation
    for feature in categorical_features:
        X[feature] = X[feature].astype(str)

    # Identify the new numeric and categorical features
    numeric_features = [
        "content_length", "count_dot_url", "count_dir_url", "count_embed_domain_url",
        "shortening_service_url", "count_http_url", "count%_url", "count?_url", 
        "count-_url", "count=_url", "url_length", "hostname_length_url", "sus_url",
        "count_digits_url", "count_letters_url", "number_of_parameters_url", 
        "number_of_fragments_url", "is_encoded_url", "special_count_url", 
        "unusual_character_ratio_url", "count_dot_content", "count_dir_content", 
        "count_embed_domain_content", "count%_content", "count?_content", 
        "count-_content", "count=_content", "sus_content", "count_digits_content", 
        "count_letters_content", "content_length", "is_encoded_content", 
        "special_count_content"
    ]

    print(f"Features generated: {numeric_features + categorical_features}")

    return X.drop(columns=["Class"]), y, numeric_features, categorical_features
