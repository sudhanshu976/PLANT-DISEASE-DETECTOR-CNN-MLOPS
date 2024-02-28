**Potato Disease Classifier - Project Readme**

### Installation
1. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2. Execute setup:
    ```bash
    pip install -e .
    ```

### Customization
- Tailor the project setup by editing `setup.py` according to your preferences.

### Project Structure
- **Logger and Exception Handling:**
  - Manage logging information and exception handling.
  - Logs stored in the 'logs' folder.

    ```bash
    /logger
    /exception
    ```

- **Main Components (`src/components`):**
  - Organized steps for data ingestion, preparation, model training, evaluation, and saving.

    ```bash
    /src
      /components
        data_ingestion.py
        data_preparation.py
        model_training.py
        model_evaluation.py
        model_saving.py
    ```

- **Training Pipeline (`src/pipeline/training_pipeline.py`):**
  - Execute with:
    ```bash
    python src/pipeline/training_pipeline.py
    ```
  - Sequentially triggers all component steps.
  - Generates 'artifacts' folder with unzipped and saved model folders.

- **Raw Data (`raw_data` folder):**
  - Contains a zip file of raw data.

    ```bash
    /raw_data
      raw_data.zip
    ```

- **Web Application (`application.py`):**
  - Flask app for the user interface.

    ```bash
    /application.py
    ```

- **HTML Template (`templates/index.html`):**
  - Enhances the look of the webpage.

    ```bash
    /templates
      /index.html
    ```

- **Static Content (`static/upload.js`):**
  - Manages image uploading on the webpage.

    ```bash
    /static
      /upload.js
    ```

### Usage
1. Run the training pipeline:
    ```bash
    python src/pipeline/training_pipeline.py
    ```
2. Launch the Flask app:
    ```bash
    python application.py
    ```
3. Access the webpage in your browser.

Feel free to customize and enhance the project as needed!
