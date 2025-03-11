# Project structure

```
.
├── dataset
│   ├── metadata.csv
│   ├── poses
│   │   ├── 20241122_122513.json
│   │   ├── 20241122_122525.json
│   │   └── ...
│   ├── processed
│   │   └── correct
│   ├── raw
│   │   ├── correct
│   │   │   ├── 20241122_122513.mp4
│   │   │   ├── 20241122_122525.mp4
│   │   │   └── ...
│   │   └── incorrect
│   │       ├── error_body_tilt
│   │       │   ├── 20241122_124943.mp4
│   │       │   ├── 20241122_124958.mp4
│   │       |   └── ...
│   │       └── error_raised_feet
│   │           ├── 20241122_123745.mp4
│   │           ├── 20241122_124052.mp4
│   │           └── ...
│   └── README.md
├── outputs
│   └── models
│       └── predictions
│           └── logs
├── README.md
├── scripts
│   ├── _prepare_video.py
│   ├── video_mirroring.py
│   ├── video_split.py
│   └── ...
├── src
│   ├── inference
│   ├── __init__.py
│   ├── models
│   ├── training
│   └── utils
└── venv
```

### Primary folders and files

- **`dataset/`**: Directory containing data, divided into `raw` (raw data) and `processed` (processed data) folders. Inside the `raw` folder, there may be subdirectories for correct and incorrect data. Each subdirectory may contain video files, such as training sessions or examples of incorrect exercise execution.
- **`outputs/`**: Directory for output generated after model processing. It includes folders for models, predictions, and logs.
- **`scripts/`**: Directory for scripts that handle data preparation tasks. All scripts are imported into the `_prepare_video.py` file and used further in the process.
- **`src/`**: Source code organized into several folders for different parts of the project, such as model training, inference execution, and utility functions.
- **`venv/`**: Directory for the virtual environment used to isolate the dependencies of the Python project.

# Requirements

- Python
- pip

### Libraries Used

`os`, `moviepy`, `MediaPipe`, `cv2`

# Preparation for work and launch
### 1. Installation **`Python`** and **`pip`**

Make sure that Python and the pip package manager are installed. You can check this by running the following commands in your terminal:

```bash
python --version
pip --version
```

If Python is not installed, you can download it from [the official Python website]([https://www.python.org/downloads/](https://www.python.org/downloads/)). Pip should be installed automatically with Python. After installation, check if pip is available. If not, install it using the following command:

```bash
python -m ensurepip --upgrade
```

### 2. Installing a virtual environment

Creating a virtual environment isolates your project's dependencies from others and system libraries. To create the environment, run the following command in the project root:

```bash
python -m venv venv
```

Next, activate the environment:

```bash
# On Linux or macOS
source venv/bin/activate

# On Windows
.\venv\Scripts\activate
```

Finally, install all dependencies with:

```bash
pip install -r requirements.txt
```
