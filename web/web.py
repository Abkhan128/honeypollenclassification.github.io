from flask import Flask, render_template, request
from PIL import Image, ImageDraw, ImageFont
import torch

app = Flask(__name__)

# Load the YOLOv7 model
model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'best.pt')
model.eval()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    # Get the uploaded image from the request
    image = request.files['image']
    img = Image.open(image)

    # Get the predictions
    results = model(img)

    # Extract the bounding boxes, labels, and scores
    boxes = results.xyxy[0].tolist()
    labels = results.names[0]
    scores = results.xyxy[0][:, 4].tolist()

    total_detections = len(boxes)

    # Count the detections for each class
    class_counts = {}
    for label in labels:
        class_counts[label] = 0

    for label_index in results.pred[0][:, -1].tolist():
        class_counts[labels[int(label_index)]] += 1

    # Calculate the percentage of detections for each class
    class_percentages = {label: count / total_detections * 100 for label, count in class_counts.items()}

    # Create the detection report
    detection_report = []
    detection_report.append("A ---> Acecia")
    detection_report.append("a ---> Wold_Flower")
    detection_report.append("c ---> Sidr ")
    detection_report.append("i ---> Trifoleum")
    detection_report.append(f"Total Detections: {total_detections}")

    for label, count in class_counts.items():
        percentage = class_percentages[label]
        detection_report.append(f"{label}: Count={count}, Percentage={percentage:.2f}%")

    # Plot the image and bounding boxes
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 40)
    my_list = [i for i in class_counts.values()]
    if my_list[0] > my_list[1] and my_list[0] > my_list[2] and my_list[0] > my_list[3]:
        detection_report.append("This honey is Acecia honey")
    elif my_list[1] > my_list[0] and my_list[1] > my_list[2] and my_list[1] > my_list[3]:
        detection_report.append("This honey is Sidr honey")
    elif my_list[3] > my_list[0] and my_list[3] > my_list[1] and my_list[3] > my_list[2]:
        detection_report.append("This honey is Trifoleum honey")

    for box, score, label_index in zip(boxes, scores, results.pred[0][:, -1].tolist()):
        xmin, ymin, xmax, ymax = box[:4]
        label = f"{labels[int(label_index)]} {score:.2f}"
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=6)
        draw.text((xmin, ymin - 40), label, font=font, fill="red")


    # Save the annotated image
    img_path = 'static/result.jpg'
    img.save(img_path)

    return render_template('report.html', img_path=img_path, detection_report=detection_report)

if __name__ == '__main__':
    app.run(debug=True)
