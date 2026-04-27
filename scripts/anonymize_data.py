"""
Anonymize OJ submission data before publishing.

Builds a SINGLE user mapping across all three dataset variants
(raw / standard / gold) so the same real user always gets the same
anonymous ID regardless of which dataset is used.

Replaces:
  - 送交者 (submitter) → user_00001 .. user_NNNNN  (consistent across files)
  - 提交ID (submission ID) → per-file sequential integers

All other columns are preserved so the dataset remains fully usable.

Usage (from repo root):
    python scripts/anonymize_data.py
"""
import csv
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')

TARGETS = [
    os.path.join(ROOT, 'submissions.csv'),
    os.path.join(ROOT, 'standard', 'submissions.csv'),
    os.path.join(ROOT, 'gold', 'submissions.csv'),
]

MAPPING_OUT = os.path.join(ROOT, 'user_mapping.json')  # keep private, do not commit


def load_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    fieldnames = list(rows[0].keys())
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_global_mapping(all_files):
    """Collect every unique username across all files, sort for determinism."""
    all_users = set()
    for path in all_files:
        if not os.path.exists(path):
            print(f'  skip (not found): {path}')
            continue
        for row in load_csv(path):
            all_users.add(row['送交者'])
    width = len(str(len(all_users)))
    return {name: f'user_{i:0{width}d}'
            for i, name in enumerate(sorted(all_users), start=1)}


def anonymize_file(path, mapping):
    if not os.path.exists(path):
        return
    rows = load_csv(path)
    for i, row in enumerate(rows, start=1):
        row['提交ID'] = str(i)
        row['送交者'] = mapping[row['送交者']]
    save_csv(path, rows)
    users_in_file = len({r['送交者'] for r in rows})
    print(f'  {os.path.relpath(path)}: {len(rows)} rows, {users_in_file} users')


def main():
    print('Building global user mapping...')
    mapping = build_global_mapping(TARGETS)
    print(f'  Total unique users across all files: {len(mapping)}')

    print('Anonymizing files...')
    for path in TARGETS:
        anonymize_file(path, mapping)

    with open(MAPPING_OUT, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f'Mapping saved to {os.path.relpath(MAPPING_OUT)} — keep private, do not commit.')


if __name__ == '__main__':
    main()
