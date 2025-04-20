class PreTrainingLayer:
    def __init__(self):
        """
        Initialize the PreTrainingLayer with necessary components
        """
        pass

    @staticmethod
    def format_value(value):
        """
        Format a value to string representation
        Args:
            value: The value to format (can be list, dict, or primitive type)
        Returns:
            str: Formatted string representation
        """
        if isinstance(value, list):
            return ', '.join(str(item) for item in value)
        return str(value)

    def format_sample_to_text(self, sample):
        """
        Convert a sample dictionary to formatted text representation
        Args:
            sample: Dictionary containing log data
        Returns:
            tuple: (formatted_text, timestamp)
        """
        text_parts = []
        timestamp_value = None

        for field, value in sample.items():
            if field.lower() == 'label':
                continue

            if field.lower() == 'timestamps':
                timestamp_value = value
                continue

            if isinstance(value, dict):
                sub_parts = []
                for sub_field, sub_value in value.items():
                    if sub_field.lower() == 'label':
                        continue
                    if sub_field.lower() == 'timestamps':
                        timestamp_value = sub_value
                        continue

                    formatted_sub_value = self.format_value(sub_value)
                    sub_parts.append(f"[{sub_field}] {formatted_sub_value}")
                text_parts.append(f"[{field}] {{{' '.join(sub_parts)}}}")
            else:
                formatted_value = self.format_value(value)
                text_parts.append(f"[{field}] {formatted_value}")

        return ' '.join(text_parts), timestamp_value

    def process_batch(self, batch_data):
        """
        Process a batch of raw data into formatted text and timestamps
        Args:
            batch_data: List of dictionaries containing raw log data
        Returns:
            tuple: (list_of_texts, list_of_timestamps, list_of_labels)
        """
        texts = []
        timestamps = []
        labels = []

        for sample in batch_data:
            text, timestamp = self.format_sample_to_text(sample)
            texts.append(text)
            timestamps.append(timestamp)
            if 'Label' in sample:
                labels.append(sample['Label'])

        return texts, timestamps, labels if labels else None