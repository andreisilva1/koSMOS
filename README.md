#  koSMOS - A playground for Machine Learning Models  
---
(currently ongoing)

## 📝 Project Purpose

**koSMOS** is designed for **rapid data exploration and machine learning prototyping**. It is **not a production ML platform**. The goal is to allow you to **test models quickly, analyze data, and generate temporary APIs**. Models live for **1 hour** before being automatically removed, making it perfect for:

- Rapid experimentation with different datasets and ML algorithms
- Understanding which models perform best on your data
- Generating temporary APIs for testing and exploration

> Ideal for learning, prototyping, and experimentation — not intended for long-term production deployments.

---

## ⚙️ Project Versions

**koSMOS** comes in **three versions**, each tailored to your workflow:

1. **Web Version** (recommended) – full interface, ideal for exploring datasets and endpoints quickly.
2. **API Version** – interact programmatically with your data and models.
3. **Local Version** – run everything on your machine for complete control.

> For web/API endpoints documentation, visit the live web version.

---

## 💻 Tech Stack

- **Python**: core programming language  
- **FastAPI**: API backend  
- **Pandas / NumPy**: data manipulation  
- **Scikit-learn**: machine learning algorithms  
- **Tailwind / HTML**: web interface  

---

## 🚀 Getting Started Locally

Follow these steps to run **koSMOS** on your machine:

### 1. Clone the repository
```bash
git clone https://github.com/andreisilva1/kosmos
cd kosmos
```

### 1.2. Create a virtual environment and activate it
```bash
python -m venv venv
source venv/bin/activate  # Linux / macOS
venv\Scripts\activate     # Windows
```

### 2. Install the dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure your environments keys
```bash
DATABASE_URL=<YOUR_MONGODB_CONNECTION_STRING>
MONGO_DB=<YOUR_MONGO_DB>
MONGO_COLLECTION=<YOUR_MONGODB_COLLECTION>
ALLOW_LOCAL_FALLBACK=1

# (EN) ALLOW_LOCAL_FALLBACK = 1 WILL USE A LOCAL SQLITE DATABASE TO SAVE AND LOAD MODELS, DEFINE AS 0 IF YOU HAVE A CONFIGURED MONGODB
```

### 4. Run the server!
```bash
cd api
uvicorn app:app --port 8001
```
**Obs:** If you choose another port, make the following change in index.html:
```bash 
index.html
const API = 'http://localhost:8001'; <- Change if necessary
```

### 5. Just open http://localhost:8001/ in the browser of your choose!
---

## ⚡ How It Works

- Upload / provide a dataset (CSV or dataframe);

- Automatic data analysis detects numerical, categorical, and ordinal features;

- Model suggestions are generated based on dataset type (classification, regression, clustering);

- Temporary models are trained and evaluated;

- APIs for models are automatically created and expire after 1 hour;

- Iteration with AI assistant recommends next models to test until performance is sufficient;
- All models and API endpoints are ephemeral to encourage rapid experimentation without long-term storage concerns.

---
## 📚 Usage Tips

- Prefer the web version for exploring data, endpoints and exclusive features (like AI analysis);

- Use the API if you want programmatic access or integration into scripts;

- The local version is best for experimenting with sensitive datasets or running offline.

---
## 🌱 Contributing

This is an active project under development. Contributions are welcome! You can help by:

- Improving model selection logic;

- Enhancing the GitHub version interface functionality;

- Adding new ML algorithms;

- Writing documentation or examples;

- Or just fixing some of the messes I made.

## 📌 Notes

- Models are ephemeral: they live for 1 hour and then are deleted.

- The system is designed for speed, not production deployment.

---
## 🔗 Links

Web interface: [https://kosmos-web.onrender.com/](https://kosmos-web.onrender.com/)

API documentation: visit the web interface

GitHub repo: [https://github.com/andreisilva1/koSMOS](https://github.com/andreisilva1/koSMOS)