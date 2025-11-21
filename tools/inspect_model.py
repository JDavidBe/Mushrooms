import joblib
import json
import sys

model_path = 'models/modelo_hongos.joblib'
try:
    m = joblib.load(model_path)
except Exception as e:
    print('ERROR_LOADING_MODEL:', e)
    sys.exit(1)

print('CLASSES:', getattr(m, 'classes_', None))
# feature_names_in_ may be numpy array
fn = getattr(m, 'feature_names_in_', None)
if fn is None:
    print('FEATURE_NAMES_IN_: None')
else:
    print('FEATURE_NAMES_COUNT:', len(fn))
    # print first 100 names for brevity
    for i, name in enumerate(fn[:200]):
        print(i, name)

# Also print intercept and coef shapes
print('INTERCEPT_SHAPE:', getattr(m, 'intercept_', None).shape)
print('COEF_SHAPE:', getattr(m, 'coef_', None).shape)
