import pandas as pd
from core.elastic import client
from core.database import engine
from export_sentence.distil_bert import extract_features

def get_data_elastic(index: str):
    query = {
        "query": {
            "match_all": {}
        }
    }
    # Bắt đầu scroll với size mỗi lần lấy (ví dụ: 1000 document) và thời gian scroll (ví dụ: 1m)
    scroll_size = 1000
    response = client.search(index=index, body=query, scroll='1m', size=scroll_size)

    # Lấy scroll_id từ response
    scroll_id = response['_scroll_id']
    data = [hit["_source"] for hit in response["hits"]["hits"]]

    # Tiếp tục scroll cho đến khi không còn dữ liệu
    while len(response["hits"]["hits"]) > 0:
        response = client.scroll(scroll_id=scroll_id, scroll='1m')
        data.extend([hit["_source"] for hit in response["hits"]["hits"]])

    # Xóa scroll khi hoàn tất
    client.clear_scroll(scroll_id=scroll_id)
    return data

def combined_sentence(document: pd.Series):
    """
    Prepare input string for DistilBERT from a DataFrame row.

    Args:
        document (pd.Series): A row of data from DataFrame.

    Returns:
        str: Input text string for the model.
    """
    analysis_id = document.get("AnalysisComponent", {}).get("AnalysisComponentIdentifier", "Unknown")
    analysis_type = document.get("AnalysisComponent", {}).get("AnalysisComponentType", "Unknown")
    component_name = document.get("AnalysisComponent", {}).get("AnalysisComponentName", "Unknown")
    message = document.get("AnalysisComponent", {}).get("Message", "No message")

    raw_logs = " ".join(document.get("LogData", {}).get("RawLogData", []))
    timestamps = ", ".join(str(ts) for ts in document.get("LogData", {}).get("Timestamps", []))
    detection_timestamp = ", ".join(str(ts) for ts in document.get("LogData", {}).get("DetectionTimestamp", []))
    log_lines_count = document.get("LogData", {}).get("LogLinesCount", "Unknown")
    log_resources = " ".join(document.get("LogData", {}).get("LogResources", []))

    aminer_id = document.get("AMiner", {}).get("ID", "Unknown")
    label = document.get("Label", "unknown")

    text_input = (
        f"[AnalysisID] {analysis_id}. "
        f"[AnalysisType] {analysis_type}. "
        f"[ComponentName] {component_name}. "
        f"[Message] {message}. "
        f"[RawLog] {raw_logs}. "
        f"[DetectionTimestamp] {detection_timestamp}. "
        f"[LogLinesCount] {log_lines_count}. "
        f"[LogResources] {log_resources}. "
        f"[IP] {aminer_id}. "
    )

    return text_input, timestamps, label

def get_dataframe(data: pd.DataFrame):
    try:
        data_import = data.copy()
        data_out = pd.DataFrame()

        # Kiểm tra kết quả trả về từ combined_sentence
        combined_results = data_import.apply(combined_sentence, axis=1)
        if not all(len(x) == 3 for x in combined_results):
            raise ValueError("combined_sentence must return exactly 3 values (text, timestamp, label)")

        data_out[["text", "timestamp", "label"]] = combined_results.tolist()

        features = data_out["text"].apply(extract_features)
        features_df = pd.DataFrame(features.tolist())

        features_df["Timestamps"] = data_out["timestamp"].values
        features_df["Label"] = data_out["label"].values
        return features_df
    except Exception as e:
        print(f"Error in get_dataframe: {str(e)}")
        raise


def export_feature_save(index_name: str):
    global scroll_id, total_samples
    if not client.ping():
        raise ConnectionError("Elasticsearch connection failed")

    if not engine:
        raise ConnectionError("Database engine not initialized")

    query = {"query": {"match_all": {}}}
    scroll_size = 1000
    scroll_timeout = '5m'  # Tăng thời gian scroll

    try:
        response = client.search(
            index=index_name,
            body=query,
            scroll=scroll_timeout,
            size=scroll_size
        )
        scroll_id = response['_scroll_id']
        total_samples = 0

        while True:
            if not response["hits"]["hits"]:
                break

            try:
                data_json = [hit["_source"] for hit in response["hits"]["hits"]]
                data_raw = pd.DataFrame(data_json)
                data = get_dataframe(data_raw)

                num_samples = data.shape[0]
                total_samples += num_samples
                print(f"Processing batch: {num_samples} samples (Total: {total_samples})")

                # Ghi vào SQL với transaction
                with engine.begin() as connection:
                    data.to_sql(
                        f"{index_name}-1",
                        con=connection,
                        schema=index_name,
                        if_exists="append",
                        index=False,
                        method='multi'
                    )
            except Exception as batch_error:
                print(f"Error processing batch: {str(batch_error)}")
                continue

            # Lấy batch tiếp theo
            try:
                response = client.scroll(scroll_id=scroll_id, scroll=scroll_timeout)
            except Exception as scroll_error:
                print(f"Scroll error: {str(scroll_error)}")
                break

    finally:
        if 'scroll_id' in locals() and scroll_id:
            try:
                client.clear_scroll(scroll_id=scroll_id)
            except Exception as clear_error:
                print(f"Error clearing scroll: {str(clear_error)}")

    return total_samples


data = get_data_elastic("aminer-fox")
print(data[0])
data_import = pd.DataFrame(data)
print(data_import.iloc[0])
print(combined_sentence(data_import.iloc[0]))