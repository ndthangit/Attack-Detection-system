from core.elastic import client
import pandas as pd

def get_data_elastic(query, index: str, scroll_size = 1000, convert_function=None):
    dataframes = []
    scroll_timeout = "5m"

    try:
        response = client.search(
            index=index,
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
                data = pd.DataFrame(data_json)

                num_samples = data.shape[0]
                total_samples += num_samples
                print(f"Processing batch: {num_samples} samples (Total: {total_samples})")

                # Lưu vào danh sách thay vì database
                dataframes.append(data)

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

    return pd.concat(dataframes, ignore_index=True) if dataframes else pd.DataFrame()