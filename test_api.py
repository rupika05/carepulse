import requests
import pandas as pd
import io

# Load 5 rows of training data (which has actual outcome)
df = pd.read_csv('data/train_ml03.csv', nrows=5)
buf = io.StringIO()
df.to_csv(buf, index=False)
csv_bytes = buf.getvalue().encode()

files = {'file': ('test.csv', csv_bytes, 'text/csv')}
r = requests.post('http://127.0.0.1:8000/batch-predict', files=files)
result = r.json()
print('total_patients:', result['total_patients'])
print('has_actuals:', result['has_actuals'])
print('distribution:', result['distribution'])
print('actual_distribution:', result['actual_distribution'])
print('accuracy:', result['accuracy'])
print('confusion_matrix:', result['confusion_matrix'])
print('num predictions:', len(result['predictions']))
if result['predictions']:
    print('sample pred:', result['predictions'][0])
