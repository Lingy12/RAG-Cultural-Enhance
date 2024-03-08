import json
import sys

# Function to extract the first character inside parentheses
def extract_char(s):
    start = s.find('(')
    end = s.find(')')
    if start != -1 and end != -1:
        return s[start + 1:end].strip()
    return None

# Function to compare two files
def compare_files(reference_file, rag_file):
    with open(reference_file, 'r') as f:
        reference_data = json.load(f)

    with open(rag_file, 'r') as f:
        rag_data = json.load(f)

    revise_count = 0
    rag_err = 0
    not_trigger_rag = 0
    correct_predictions = {"RAG": 0, "Baseline": 0}
    i = 0
    for ref, rag in zip(reference_data, rag_data):
        ref_pred = extract_char(ref['model_prediction'])
        ref_ans = extract_char(ref['answer'])
        rag_pred = extract_char(rag['model_prediction'])
        rag_ans = extract_char(rag['answer'])

        if rag['model_prediction'].endswith('[RAG]'):
            if rag_pred == rag_ans and ref_pred != ref_ans:
                revise_count += 1
                print('RAG_REVISE'.center(50, '='))
                print('RAG')
                print(json.dumps(rag_data[i], indent=4))
                print('BASELINE')
                print(json.dumps(reference_data[i], indent=4))
            elif rag_pred != rag_ans and ref_pred == ref_ans:
                rag_err += 1
                print('RAG_ERROR'.center(50, '*'))
                print('RAG')
                print(json.dumps(rag_data[i], indent=4))
                print('BASELINE')
                print(json.dumps(reference_data[i], indent=4))

        else:
            not_trigger_rag += 1

        if rag_pred == rag_ans:
            correct_predictions['RAG'] += 1

        if ref_ans == ref_pred:
            correct_predictions['Baseline'] += 1
        i += 1
    return {
        'revise_count': revise_count,
        'rag_err': rag_err,
        'not_trigger_rag': not_trigger_rag,
        'correct_predictions': correct_predictions
    }

if __name__ == '__main__':
    # Example usage
    reference_file = sys.argv[1]
    rag_file = sys.argv[2]
    result = compare_files(reference_file, rag_file)
    print('Final result'.center(50, '='))
    print(json.dumps(result, indent=4))
