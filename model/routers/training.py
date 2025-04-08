from operator import index
from fastapi import APIRouter
from core.elastic import client
from purposed_model.custom_PWAGAT import custom_PWAGAT
import pandas as pd

router = APIRouter()


def format_sample_to_text(sample):
    text_parts = []
    timestamp_value = None

    for field, value in sample.items():
        if field.lower() == 'label':
            continue

        if field.lower() == 'timestamps':
            timestamp_value = value
            continue

        if isinstance(value, dict):
            # Xử lý trường hợp value là dictionary (trường lớn có trường con)
            sub_parts = []
            for sub_field, sub_value in value.items():
                if sub_field.lower() == 'label':
                    continue
                if sub_field.lower() == 'timestamps':
                    timestamp_value = sub_value
                    continue

                formatted_sub_value = format_value(sub_value)
                sub_parts.append(f"[{sub_field}] {formatted_sub_value}")
            text_parts.append(f"[{field}] {{{' '.join(sub_parts)}}}")
        else:
            # Xử lý trường hợp value không phải dictionary
            formatted_value = format_value(value)
            text_parts.append(f"[{field}] {formatted_value}")

    return ' '.join(text_parts), timestamp_value

def format_value(value):
    if isinstance(value, list):
        # Xử lý trường hợp value là list, join bằng dấu ,
        return ', '.join(str(item) for item in value)
    else:
        return str(value)

@router.get("/", tags=["train"])
async def training():
    global scroll_id

    input_size = 768
    hidden_size_g = 31
    hidden_size_d = 42
    latent_size = 128
    output_size = 768
    dropout_rate_g = 0.42
    dropout_rate_d = 0.32
    learning_rate_g = 0.008
    learning_rate_d = 0.009
    num_epochs_g = 25  # 152

    hidden_size_mlp = 53
    dropout_rate_mlp = 0.31
    learning_rate_mlp = 0.005
    num_episodes = 25  # 245
    batch_size_mlp = 74  # Đồng bộ với giá trị đã xác định trước đó
    batch_size_rl = batch_size_mlp  # Đồng bộ batch_size_rl với batch_size_mlp

    model = custom_PWAGAT(
        input_size=input_size,
        hidden_size_g=hidden_size_g,
        hidden_size_d=hidden_size_d,
        dropout_rate_g=dropout_rate_g,
        dropout_rate_d=dropout_rate_d,
        learning_rate_g=learning_rate_g,
        learning_rate_d=learning_rate_d,
        num_epochs_g=num_epochs_g,
        hidden_size_mlp=hidden_size_mlp ,
        dropout_rate_mlp=dropout_rate_mlp,
        learning_rate_mlp=learning_rate_mlp,
        num_episodes=num_episodes,
        batch_size_mlp=batch_size_mlp,
        seq_len=10
    )
    # Lấy danh sách tất cả index
    response = client.cat.indices(index="*", h="index", format="json")

    # print("Các index người dùng (không phải hệ thống):")
    # for name in user_indices:
    #     print(name)

    user_indices = ['aminer-fox']

    for index_name in user_indices:
        print(f"Đang xử lý index: {index_name}")
        query = {
            "query": {
                "match_all": {}
            }

        }
        scroll_size = 100
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
                    # print(data_json)
                    data_raw = pd.DataFrame(data_json)

                    # Áp dụng format_sample_to_text để tạo hai cột
                    text_and_timestamps = data_raw.apply(lambda row: format_sample_to_text(row), axis=1)
                    data_raw['text_representation'] = text_and_timestamps.apply(lambda x: x[0])
                    data_raw['timestamps'] = text_and_timestamps.apply(lambda x: x[1])

                    # Nếu bạn muốn xem kết quả
                    # print(data_raw['text_representation'].head())
                    print(data_raw.columns)
                    print(data_raw.head())
                    model.fit(data_raw['text_representation'],data_raw['timestamps'], data_raw['Label'])

                    # print(f"Processing batch: {num_samples} samples (Total: {total_samples})")

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

    return {"message": "Training completed successfully."}
