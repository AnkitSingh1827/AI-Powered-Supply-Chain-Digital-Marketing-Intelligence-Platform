IDENTIFIER_COLUMNS = [
    "Shipment_ID",
    "Campaign_ID",
]


def remove_identifier_columns(df, identifiers=None):
    identifiers = IDENTIFIER_COLUMNS if identifiers is None else identifiers
    return df.drop(
        columns=[c for c in identifiers if c in df.columns],
        errors="ignore",
    )


def get_numeric_columns(df):
    return df.select_dtypes(include=["number"]).columns.tolist()


def get_categorical_columns(df):
    return df.select_dtypes(
        include=["object", "category", "string"]
    ).columns.tolist()
