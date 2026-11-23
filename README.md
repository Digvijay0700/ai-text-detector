# AI Text Detector

A Python project that trains and compares two human-vs-AI text classifiers, evaluates their results, and serves the transformer model in a Streamlit app.

## Open and run in VS Code

1. Clone this repository and open the `ai-text-detector` folder in VS Code.
2. Open **Terminal → New Terminal** in VS Code.
3. Create and activate a virtual environment:

   **macOS/Linux**

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   **Windows PowerShell**

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

4. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. Run the complete pipeline from the project root:

   ```bash
   python -m src.data
   python -m src.train_baseline
   python -m src.train_transformer
   python -m src.evaluate
   streamlit run app.py
   ```

The first command downloads the Hugging Face dataset and creates train, validation, and test splits under `data/`. Training creates model files under `models/`; evaluation writes metrics, plots, and failure examples under `report/`. These generated files are kept out of Git. The transformer training step downloads the base model and can take a while; it uses an available accelerator when supported and otherwise runs on the CPU.

To launch the app again after training, activate the virtual environment and run `streamlit run app.py` from the project root.
