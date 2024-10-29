# Hurricane Preview

This project visualizes hurricane data using `matplotlib` and `seaborn`.

## Setup Instructions

### Prerequisites

- Python 3.x installed on your machine
- `pip` (Python package installer)

### Setting Up a Virtual Environment

1. **Create a virtual environment**:
   
   Open a terminal and navigate to the project directory. Run the following command to create a virtual environment named `.venv`:

   ```bash
   python3 -m venv .venv
   
2. **Activate the virtual environment**:

    On macOS and Linux:
    ```bash
    source .venv/bin/activate
    ```
    On Windows:
    ```
    .venv\Scripts\activate
    ```

3. **Install the required packages**:

    With the virtual environment activated, install the necessary packages using pip:
    ```
    pip install -r requirements.txt
    ```

### Running the Script

With the virtual environment activated, you can run the hurricane_preview.py script:


    python hurricane_preview.py


### Dataset

This project uses the **[Benchmark Dataset for Automatic Damaged Building Detection from Post-Hurricane Remotely Sensed Imagery]https://drive.google.com/drive/folders/1NvYoKQ8_oxA7oeCN5alwpLiFOC19_OF7?usp=sharing** for training the model.

Youngjun Choe, Valentina Staneva, Tessa Schneider, Andrew Escay, Christopher Haberland, Sean Chen, December 12, 2018, "Benchmark Dataset for Automatic Damaged Building Detection from Post-Hurricane Remotely Sensed Imagery", IEEE Dataport, doi: https://dx.doi.org/10.21227/1s3n-f891.


### Deactivating the Virtual Environment

When you're done, you can deactivate the virtual environment by running:


    deactivate


