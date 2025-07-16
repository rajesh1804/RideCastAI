---
title: "RideCastAI"
emoji: "🚖 "
colorFrom: "blue"
colorTo: "green"
sdk: streamlit
sdk_version: "1.32.2"
app_file: streamlit_app/app.py
pinned: false
---

# RideCastAI

```bash
(venv) D:\Projects\RideCastAI>python src\train_eta_model.py 
📥 Loading processed data...
🔀 Splitting into train/test...
🚀 Starting ETA model training...
✅ Training completed in 3.75 seconds   
📈 Evaluating model on test set...
📊 ETA MAE: 3.11 minutes
💾 ETA model saved to src/models/eta_model.pkl
```

```bash
(venv) D:\Projects\RideCastAI>python src\train_fare_model.py
🔄 Loading and preprocessing data...
✅ Data loaded and split. Time taken: 5.58s

🚀 Training quantile model for q=0.1...
✅ Done. MAE @ quantile 0.1: 2.72 | Time taken: 441.53s
💾 Model saved to src/models/fare_model_q10.pkl

🚀 Training quantile model for q=0.5...
✅ Done. MAE @ quantile 0.5: 1.85 | Time taken: 483.59s
💾 Model saved to src/models/fare_model_q50.pkl

🚀 Training quantile model for q=0.9...
✅ Done. MAE @ quantile 0.9: 3.44 | Time taken: 442.07s
💾 Model saved to src/models/fare_model_q90.pkl

🎉 All quantile models trained and saved successfully.
```