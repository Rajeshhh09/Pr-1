# Computer Vision Pipeline

A hands-on computer vision project that starts with simple image operations and builds up to detecting faces and everyday objects. The work is collected in the Jupyter notebook [`CV_PR3.ipynb`](./CV_PR3.ipynb).

> **In simple terms:** the project cleans and studies images, finds faces with YuNet, finds objects with YOLOv8, and compares how quickly the methods run.

## What you will explore

1. **Image processing** — load and display images, convert them to grayscale, and try thresholding and morphological operations such as erosion, dilation, opening, and closing.
2. **Image analysis** — use masks and bitwise operations, inspect colour histograms, and adjust brightness and contrast.
3. **Face detection with YuNet** — find faces and facial landmarks in photos, test confidence thresholds, and try live webcam detection.
4. **Object detection with YOLOv8** — detect common objects in photos, explore confidence and IoU thresholds, and try live webcam detection.
5. **Combined pipeline and comparison** — run YuNet and YOLOv8 together, compare their speed, and summarize the techniques.


## Project files

```text
CV_PR3/
├── CV_PR3.ipynb
├── yolov8n.pt
├── data/
│   ├── images/
│   │   ├── bottles.jpg
│   │   ├── cars.jpeg
│   │   ├── Chairs.jpeg
│   │   ├── face_profile.jpeg
│   │   ├── face_sunglasses.jpeg
│   │   ├── my_photo.jpg
│   │   ├── persons.jpeg
│   │   ├── solid dice.jpg
│   │   └── solid pot.jpg
│   └── models/
│       └── face_detection_yunet_2023mar.onnx
└── plots/
    ├── face_detection_comparison.png
    ├── fps_comparison.png
    ├── results_table.png
    ├── yolo_class_summary.png
    └── yolo_detections.png
```

#  Video Link :- https://drive.google.com/file/d/1Kizy8hlj-giOu98bY6Yxx7iEMFQVFoGr/view?usp=sharing 

The notebook expects to be run with this project folder as its working directory. It uses the image and model paths shown above.

## Get started

You need Python and Jupyter Notebook support. From PowerShell, open the project folder and create an environment:

```powershell
cd "D:\Machine Learning\Deep Learning\Pr 1\CV_PR3"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install opencv-contrib-python numpy matplotlib pandas ultralytics ipykernel
```

Then open `CV_PR3.ipynb` in VS Code or Jupyter and select the Python kernel from this environment. Run the cells from the top in order so that imports, images, and models are ready before later cells use them.

**OpenCV note:** install only one OpenCV package in the environment. Do not install `opencv-python` and `opencv-contrib-python` together. If using the notebook's install cell, replace its command with:

```python
%pip install opencv-contrib-python numpy matplotlib pandas ultralytics
```

Restart the notebook kernel after installing packages if it was already running.

## Images and models

- **YuNet model:** `data/models/face_detection_yunet_2023mar.onnx`
- **YOLOv8 model:** `yolov8n.pt`
- **Face examples:** `my_photo.jpg`, `persons.jpeg`, `face_profile.jpeg`, and `face_sunglasses.jpeg`
- **Object examples:** `cars.jpeg`, `bottles.jpg`, and `Chairs.jpeg`

The notebook uses YuNet for faces and YOLOv8 Nano for general objects. Detection depends on the input image: for example, YuNet is meant for face images, not `solid dice.jpg`.

## Running the webcam demos

The webcam cells need a connected camera and permission to use it. Run those cells when you are ready to use the camera, and press **Q** in the video window to stop the live demo. If you do not have a camera, skip the webcam and webcam-benchmark cells; the photo-based examples can still be run.

## Outputs

The notebook displays results inline and saves visual summaries in `plots/`, including:

- Face-detection examples
- YOLOv8 detections and per-class counts
- Frames-per-second comparison
- Final technique comparison table

Re-run the relevant notebook cells to refresh their outputs.

## Quick troubleshooting

| Problem | What to check |
| --- | --- |
| An image or model cannot be found | Start Jupyter from the `CV_PR3` project folder and check that the expected files exist under `data/`. |
| YuNet shows no face detections | Use one of the face example images, confirm the YuNet ONNX file exists, and try a lower score threshold such as `0.5`. |
| A detection-count chart has no visible bars | Check the printed data first. If every detection count is `0`, the chart is plotting the results correctly, but the detector found no matching objects in that image. |
| `cv2` is unavailable or behaves unexpectedly | Confirm the notebook is using the environment where OpenCV was installed. Install only one OpenCV package, then restart the kernel. |
| The webcam will not open | Check camera permissions, close other apps using the camera, or skip the webcam cells and use the static-image examples. |
| A plot or image does not save | Check that the project folder is writable and that the `plots/` folder exists. |

## Libraries used

- [OpenCV](https://opencv.org/) — image processing and YuNet face detection
- [NumPy](https://numpy.org/) — numerical operations
- [Matplotlib](https://matplotlib.org/) — charts and image display
- [pandas](https://pandas.pydata.org/) — result tables
- [Ultralytics YOLO](https://docs.ultralytics.com/) — YOLOv8 object detection

