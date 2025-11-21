import joblib
import pandas as pd

model = joblib.load('models/modelo_hongos.joblib')
model_features = model.feature_names_in_

# Two sample inputs with different odors and gill colors
samples = [
    {
        'bruises': 't',
        'gill-color': 'y',
        'gill-size': 'b',
        'gill-spacing': 'c',
        'habitat': 'd',
        'odor': 'n',
        'population': 's',
        'ring-type': 'n',
        'spore-print-color': 'w',
        'stalk-color-above-ring': 'n',
        'stalk-color-below-ring': 'n',
        'stalk-root': 'b',
        'stalk-surface-above-ring': 's',
        'stalk-surface-below-ring': 's'
    },
    {
        'bruises': 'f',
        'gill-color': 'k',
        'gill-size': 'n',
        'gill-spacing': 'w',
        'habitat': 'g',
        'odor': 'p',
        'population': 'a',
        'ring-type': 'p',
        'spore-print-color': 'k',
        'stalk-color-above-ring': 'p',
        'stalk-color-below-ring': 'p',
        'stalk-root': '?',
        'stalk-surface-above-ring': 'f',
        'stalk-surface-below-ring': 'f'
    }
]

for i, s in enumerate(samples):
    # Build DataFrame like app does
    X_input = pd.DataFrame({k: [v] for k, v in s.items()})
    X_encoded = pd.get_dummies(X_input, drop_first=False)
    X_final = pd.DataFrame(0, index=[0], columns=model_features)
    for col in X_encoded.columns:
        if col in model_features:
            X_final[col] = X_encoded[col].values[0]
    # ensure no NaNs
    for feature in model_features:
        if feature not in X_final.columns or X_final[feature].isna().any():
            X_final[feature] = 0
    proba = model.predict_proba(X_final)[0]
    pred = model.predict(X_final)[0]
    print(f"Sample {i} pred={pred} proba={proba}")
    # print non-zero columns
    nz = [c for c in X_final.columns if X_final.loc[0, c] != 0]
    print('Non-zero features:', nz)
    print('---')
