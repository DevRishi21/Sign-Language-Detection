# Sign Language Translator

Recognize American Sign Language (ASL) fingerspelled letters from images using a fine-tuned ResNet50, then translate the recognized text into multiple languages with NLLB-200. The result is shown in a chat-style Streamlit app that accepts one or more uploaded images.

> project (Image-Based Sign Language Recognition to Text and Multilingual Translation Using Deep Learning).

---

## Features

- **Letter recognition** for 24 static ASL letters (A–I, K–Y; J and Z need motion and are excluded)
- **Transfer learning** with ResNet50 and EfficientNetB0, both trained in two phases (frozen head, then fine-tuning)
- **Hand localization** with MediaPipe Hands (Tasks API) to crop uploaded photos to the hand
- **Multilingual translation** with NLLB-200 (distilled 600M) to French, Spanish, German, Hindi, and Tamil
- **Chat-style interface** with per-letter confidence, low-confidence warnings, and top-3 alternatives

## How it works

```
Uploaded image(s)
      │
      ▼
Limit size ──► Hand crop (MediaPipe) ──► Grayscale, pad to square, 28×28
      │
      ▼
ResNet50 (fine-tuned, 224×224 input) ──► Letter + confidence per image
      │
      ▼
NLLB-200 ──► Translations
```

## Results

Evaluated on a stratified 20% validation split of Sign Language MNIST (5,491 images).

| Model | Validation accuracy | Macro F1 | Parameters | Inference (GPU, per image) |
|---|---|---|---|---|
| EfficientNetB0 (fine-tuned) | 0.9976 | 0.9976 | 4.04M | 1.35 ms |
| **ResNet50 (fine-tuned)** | **0.9989** | **0.9989** | 23.56M | 2.95 ms |

ResNet50 was selected for the app because it made about half as many errors. Both models run well within real-time limits.

Translation was verified qualitatively on sample words and sentences in all five target languages.

### Important caveats
- The model is trained on 28×28 grayscale images. Accuracy on uncontrolled real photos is lower than on the dataset, and high-resolution photos depend on the hand crop working.

## Dataset

- **Sign Language MNIST** (training CSV): 27,455 images, 784 pixels each (28×28 grayscale), labels 0–24 with 9 (J) absent.
- Labels are remapped to contiguous indices 0–23. The mapping to letters is saved in `models/labels.json`.

The dataset is not included in this repository. Download `sign_mnist_train.csv` from Kaggle and place it in `data/`.

## Project structure

```
sign-language-translator/
├── app/
│   └── streamlit_app.py          # Chat-style Streamlit interface
├── data/                         # Dataset (not committed)
├── models/
│   ├── resnet50_sign_classifier.pt   # Final model + metadata
│   └── labels.json          # Class index → letter mapping
├── notebooks/
│   └── model_training.ipynb      # EDA, preprocessing, training, evaluation
├── src/
│   ├── inference/
│   │   ├── preprocess.py         # Size limiting and MediaPipe hand cropping
│   │   └── predictor.py          # Image → letter, with confidence
│   └── translation/
│       └── translator.py         # NLLB-200 wrapper
├── test_images/                  # Sample images for testing
├── requirements.txt
└── README.md
```

## Running the app

From the project root:

```bash
streamlit run app/streamlit_app.py
```

1. Upload one or more sign images, in the order you want them read.
2. Click **Recognize and translate**.
3. Choose target languages in the sidebar.

Each image is one letter. Upload several images to form a word; a single letter is shown but not translated.

## Training the models

Open `notebooks/model_training.ipynb` and run the cells in order:

1. Load and inspect the dataset (EDA)
2. Preprocess: reshape, normalize, stratified 80/20 split, DataLoaders
3. Train EfficientNetB0 and ResNet50 (phase 1: head only; phase 2: fine-tune the last blocks)
4. Evaluate per class (precision, recall, F1, confusion matrix)
5. Save the final model and label mapping

A GPU is strongly recommended. Training ResNet50 at 224×224 on CPU is slow.

## Limitations

- **Real photos:** The model was trained on controlled Sign MNIST images. Accuracy drops on photos with different lighting, backgrounds, camera angles, or framing. Hand cropping helps, but it does not remove this gap.
- **Resolution:** Inputs are downscaled to 28×28, which loses fine finger detail (for example, the ring shape of F).
- **Static letters only:** J and Z require motion and are not recognized. Continuous signing and words signed with motion are out of scope.
- **Word breaks:** Spaces are not predicted by the model. They come from automatic splitting, or from a blank separator image if that option is enabled. Names and uncommon words may split incorrectly.
- **Translation quality:** Verified qualitatively. A BLEU/chrF evaluation on reference sentences is planned.

## Acknowledgements

- Sign Language MNIST dataset (Kaggle)
- ResNet and EfficientNet pretrained weights (torchvision)
- NLLB-200 by Meta AI
- MediaPipe Hands by Google
