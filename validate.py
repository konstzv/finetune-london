import json
import sys

def validate_file(filepath):
    errors = []
    total = 0
    categories = {}

    with open(filepath) as f:
        for i, line in enumerate(f, 1):
            total += 1
            line = line.strip()
            if not line:
                errors.append(f"Line {i}: empty line")
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON - {e}")
                continue

            if "messages" not in obj:
                errors.append(f"Line {i}: missing 'messages' key")
                continue

            messages = obj["messages"]
            roles = [m.get("role") for m in messages]

            for required in ["system", "user", "assistant"]:
                if required not in roles:
                    errors.append(f"Line {i}: missing role '{required}'")

            for m in messages:
                if not m.get("content", "").strip():
                    errors.append(f"Line {i}: empty content for role '{m.get('role')}'")

            assistant_msg = next((m for m in messages if m["role"] == "assistant"), None)
            if assistant_msg:
                cat = assistant_msg["content"]
                categories[cat] = categories.get(cat, 0) + 1

    return total, errors, categories


files = sys.argv[1:] or ["train.jsonl", "eval.jsonl"]

all_passed = True
for filepath in files:
    print(f"\n=== {filepath} ===")
    total, errors, categories = validate_file(filepath)

    if errors:
        all_passed = False
        for e in errors:
            print(f"  ERROR: {e}")
    else:
        print(f"  All {total} lines valid")

    print(f"  Categories: {categories}")

print()
if all_passed:
    print("RESULT: ALL VALID")
else:
    print("RESULT: ERRORS FOUND")
    sys.exit(1)
