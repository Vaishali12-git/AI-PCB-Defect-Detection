
import os
import threading
import spaces
import gradio as gr
import pandas as pd
from PIL import Image
from ultralytics import YOLO

# ----------------------------------
# 1. MODEL PATHS
# ----------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

model_paths = {
    "Electronic Component Detection":
        os.path.join(MODEL_DIR, "component_model.pt"),

    "PCB Surface Defect Detection":
        os.path.join(MODEL_DIR, "pcb_model.pt"),

    "Solder & Assembly Defect Detection":
        os.path.join(MODEL_DIR, "solder_model.pt")
}

# ----------------------------------
# 2. LOAD TRAINED MODELS
# ----------------------------------

models = {}

for name, path in model_paths.items():

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    print("Loading:", name)

    model = YOLO(path)

    # Place model on ZeroGPU-compatible CUDA device
    model.to("cuda")

    models[name] = model

print("All three trained models loaded successfully!")

# Prevent simultaneous inference calls from using shared models
inference_lock = threading.Lock()

# ----------------------------------
# 3. DETECTION FUNCTION
# ----------------------------------

@spaces.GPU(duration=30)
def detect_objects(image, selected_model, confidence):

    if image is None:
        return None, "Please upload an image.", pd.DataFrame()

    try:
        with inference_lock:

            model = models[selected_model]

            results = model.predict(
                source=image,
                conf=confidence,
                imgsz=640,
                iou=0.45,
                max_det=100,
                verbose=False
            )

            result = results[0]

            annotated = result.plot()
            annotated = Image.fromarray(annotated[:, :, ::-1])

            detections = []

            if result.boxes is not None:

                for box in result.boxes:

                    class_id = int(box.cls[0])
                    score = float(box.conf[0])

                    detections.append({
                        "Detected Class": model.names[class_id],
                        "Confidence (%)": round(score * 100, 2)
                    })

            df = pd.DataFrame(
                detections,
                columns=["Detected Class", "Confidence (%)"]
            )

            if detections:
                summary = (
                    "Detection completed successfully!\n"
                    f"Selected module: {selected_model}\n"
                    f"Total detections: {len(detections)}"
                )
            else:
                summary = (
                    "No objects or defects detected.\n"
                    "Try another suitable image or confidence threshold."
                )

            return annotated, summary, df

    except Exception as error:
        return None, f"Detection error: {error}", pd.DataFrame()

# ----------------------------------
# 4. GRADIO DASHBOARD
# ----------------------------------

with gr.Blocks(
    title="AI PCB Intelligence System",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # AI PCB Intelligence System

        ### AI-Powered Electronic Component and PCB Defect Detection

        An integrated AI application for electronic component,
        PCB surface defect, and solder assembly defect detection.
        """
    )

    with gr.Row():

        with gr.Column():

            input_image = gr.Image(
                type="pil",
                label="Upload Electronic / PCB Image"
            )

            model_choice = gr.Dropdown(
                choices=list(models.keys()),
                value="Electronic Component Detection",
                label="Select Detection Module"
            )

            confidence = gr.Slider(
                minimum=0.10,
                maximum=0.90,
                value=0.25,
                step=0.05,
                label="Confidence Threshold"
            )

            detect_button = gr.Button(
                "Start AI Detection",
                variant="primary"
            )

        with gr.Column():

            output_image = gr.Image(label="Detection Result")

            output_summary = gr.Textbox(
                label="Detection Summary",
                lines=4
            )

            output_table = gr.Dataframe(
                headers=["Detected Class", "Confidence (%)"],
                label="Detected Objects and Confidence",
                interactive=False
            )

    detect_button.click(
        fn=detect_objects,
        inputs=[input_image, model_choice, confidence],
        outputs=[output_image, output_summary, output_table]
    )

    gr.Markdown(
        """
        ---
        **AI PCB Intelligence System | Three Integrated Detection Models**
        """
    )

# ----------------------------------
# 5. APPLICATION ENTRY POINT
# ----------------------------------

if __name__ == "__main__":
    demo.launch()
