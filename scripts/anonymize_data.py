"""
Anonymize OJ submission data before publishing.

Builds consistent mappings across all dataset variants (raw/standard/gold):
  - 送交者  → user_00001 .. user_NNNNN
  - 题目     → prob_0001  .. prob_MMMM   (same ID in CSV and problems.json)
  - 题目链接 → /problem/prob_XXXX/

Submission IDs are re-sequenced per file.
All other columns/fields are preserved.

Usage (from repo root):
    python scripts/anonymize_data.py
"""
import csv
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

CSV_FILES = [
    os.path.join(ROOT, 'submissions.csv'),
    os.path.join(ROOT, 'standard', 'submissions.csv'),
    os.path.join(ROOT, 'gold', 'submissions.csv'),
]

JSON_FILES = [
    os.path.join(ROOT, 'problems.json'),
    os.path.join(ROOT, 'standard', 'problems.json'),
    os.path.join(ROOT, 'gold', 'problems.json'),
]

MAPPING_OUT = os.path.join(ROOT, 'anon_mapping.json')  # keep private, do not commit


def load_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_mapping(all_values, prefix, width):
    """Sort values for determinism, assign anonymous IDs."""
    return {v: f'{prefix}{i:0{width}d}'
            for i, v in enumerate(sorted(all_values), start=1)}


def collect_users(csv_files):
    users = set()
    for path in csv_files:
        if not os.path.exists(path):
            continue
        for row in load_csv(path):
            users.add(row['送交者'])
    return users


def collect_problems(csv_files, json_files):
    problems = set()
    for path in csv_files:
        if not os.path.exists(path):
            continue
        for row in load_csv(path):
            problems.add(row['题目'])
    for path in json_files:
        if not os.path.exists(path):
            continue
        with open(path, encoding='utf-8') as f:
            for entry in json.load(f):
                problems.add(entry['题目'])
    return problems


def anonymize_csv(path, user_map, prob_map):
    if not os.path.exists(path):
        return
    rows = load_csv(path)
    for i, row in enumerate(rows, start=1):
        row['提交ID'] = str(i)
        row['送交者'] = user_map[row['送交者']]
        anon_prob = prob_map[row['题目']]
        row['题目'] = anon_prob
        row['题目链接'] = f'/problem/{anon_prob}/'
    save_csv(path, rows)
    print(f'  {os.path.relpath(path)}: {len(rows)} rows')


def anonymize_json(path, prob_map):
    if not os.path.exists(path):
        return
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    for entry in data:
        entry['题目'] = prob_map[entry['题目']]
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f'  {os.path.relpath(path)}: {len(data)} entries')


def main():
    print('Collecting all unique users and problems...')
    users = collect_users(CSV_FILES)
    problems = collect_problems(CSV_FILES, JSON_FILES)
    print(f'  Users: {len(users)}, Problems: {len(problems)}')

    user_map = build_mapping(users, 'user_', len(str(len(users))))
    prob_map = build_mapping(problems, 'prob_', len(str(len(problems))))

    print('Anonymizing CSV files...')
    for path in CSV_FILES:
        anonymize_csv(path, user_map, prob_map)

    print('Anonymizing problems.json files...')
    for path in JSON_FILES:
        anonymize_json(path, prob_map)

    mapping = {'users': user_map, 'problems': prob_map}
    with open(MAPPING_OUT, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f'Mappings saved to {os.path.relpath(MAPPING_OUT)} — keep private, do not commit.')


if __name__ == '__main__':
    main()
