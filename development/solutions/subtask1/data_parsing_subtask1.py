import csv

def load_data(file_path):
    data = []
    labels = []
    with open(file_path, encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) < 5:
                continue
            article_id, entity_mention, start_offset, end_offset, main_role, *fine_grained_roles = row
            data.append((article_id, entity_mention, start_offset, end_offset))
            labels.append(fine_grained_roles)
    return data, labels

def extract_context(article_path, start_offset, end_offset, window=100):
    with open(article_path, encoding="utf-8") as f:
        content = f.read()
    start_offset = max(0, int(start_offset) - window)
    end_offset = min(len(content), int(end_offset) + window)
    return content[start_offset:end_offset]