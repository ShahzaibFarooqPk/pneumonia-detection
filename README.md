# Pneumonia Detection from Chest X-rays

Deep learning project that classifies chest X-ray images as **NORMAL** or **PNEUMONIA**, built in three stages. Each stage has its own branch.

## Project Stages

| Branch | Description |
|---|---|
| `cnn-vgg16` (builds on `main`) | Custom CNN trained from scratch (`Pneumonia_Custom_CNN.ipynb`), then improved with VGG16 transfer learning (`Pneumonia_CNN_VGG16.ipynb`) |
| `cnn-vgg16-streamlit` | Saved model deployed as a Streamlit web app (`Pneumonia_CNN_VGG16_Streamlit/`) |

## Repository Structure

```
Pneumonia_Detection/
├── Pneumonia_Custom_CNN.ipynb
├── Pneumonia_CNN_VGG16.ipynb
└── Pneumonia_CNN_VGG16_Streamlit/
    ├── app.py
    ├── Pneomonia_Application.ipynb
    ├── custom_pre_trained_model_10.keras
    └── normal_img_pneomonia.jpg
```

## Run the Streamlit App

```bash
pip install streamlit tensorflow pillow numpy
cd Pneumonia_CNN_VGG16_Streamlit
streamlit run app.py
```

Upload a chest X-ray (jpg, jpeg or png) and click **Predict**.

## Tech Stack

Python, TensorFlow/Keras, VGG16, Streamlit, NumPy, Pillow

## Disclaimer

Educational project only. The model output is not a medical diagnosis.

## Author

Muhammad Shahzaib Farooq