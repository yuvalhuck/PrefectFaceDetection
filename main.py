
from prefect import flow, task
from prefect.tasks import exponential_backoff
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import uuid
import torch
from facenet_pytorch import MTCNN
from PIL import Image
import pymongo
import shutil
import os

MONGODB_CONNECTION_STRING = "mongodb://localhost:27017/"
IMAGES_DIRECTORY = "downloaded_images"
CONVERTED_IMAGE_SIZE = (800, 600)
MINIMAL_PEOPLE_DETECTION_CONFIDENCE = 0.8

people_detection_model = None
face_detection_model = None


@task(log_prints=True, retries=3, retry_delay_seconds=exponential_backoff(backoff_factor=3))
def download_image(image_url, images_location):
    """
    Download an image from a given URL and save it to the specified location
    """
    response = requests.get(image_url)
    response.raise_for_status()

    file_name = image_url.split("/")[-1].split("?")[0]
    file_path = os.path.join(images_location, file_name)
    with open(file_path, 'wb') as file:
        file.write(response.content)

    print(f"Downloaded {image_url}")

    return file_path


@task(log_prints=True, retries=3, retry_delay_seconds=exponential_backoff(backoff_factor=3))
def list_images_in_url(url):
    """
    Scrape images from a given URL
    """
    response = requests.get(url)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')
    images = soup.find_all('img')

    images_url = []
    for img in images:
        img_url = img.get('src')
        if img_url:
            absolute_img_url = urljoin(url, img_url)
            images_url.append(absolute_img_url)

    print(f"Found {len(images_url)} images in {url}")

    return images_url


@task(log_prints=True)
def convert_image(image_path):
    """
    Convert an image to JPEG format and resize it to a specific size
    """
    with Image.open(image_path) as img:
        # Convert to JPEG
        img = img.convert("RGB")

        # Calculate the size and position for cropping or padding
        width, height = img.size
        target_width, target_height = CONVERTED_IMAGE_SIZE
        ratio = min(target_width / width, target_height / height)
        new_size = (int(width * ratio), int(height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

        # Create a new image with white background and paste the resized image
        new_img = Image.new("RGB", CONVERTED_IMAGE_SIZE, (255, 255, 255))
        new_img.paste(img, ((CONVERTED_IMAGE_SIZE[0] - new_size[0]) // 2, (CONVERTED_IMAGE_SIZE[1] - new_size[1]) // 2))

        converted_image_path = image_path.split(".")[0] + ".jpg"
        new_img.save(converted_image_path, "JPEG")

    return converted_image_path


@task(log_prints=True)
def detect_people(image_path):
    """
    Detect people in the image
    """
    global people_detection_model

    if people_detection_model is None:
        people_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

    results = people_detection_model([image_path])

    detected_people = []
    for detection in results.xyxy[0]:
        x_min = detection[0].item()
        y_min = detection[1].item()
        x_max = detection[2].item()
        y_max = detection[3].item()
        confidence = detection[4].item()
        class_id = int(detection[5].item())  # class_id should be an integer

        if class_id != 0:  # 0 is the class ID for person
            continue

        if confidence < MINIMAL_PEOPLE_DETECTION_CONFIDENCE:
            continue

        detected_people.append((x_min, y_min, x_max, y_max))

    print(f"Detected {len(detected_people)} people in {image_path}")

    return detected_people


@task(log_prints=True)
def detect_faces(image_path):
    """
    Detect faces in the image
    """
    global face_detection_model

    if face_detection_model is None:
        face_detection_model = MTCNN(keep_all=True, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    image = Image.open(image_path)
    boxes, _ = face_detection_model.detect(image)

    detected_faces = boxes.tolist() if boxes is not None else []

    print(f"Detected {len(detected_faces)} faces in {image_path}")

    return detected_faces


@task(log_prints=True)
def assign_faces_to_people(people_xyxy, faces_xyxy):
    """
    Assign detected faces to detected people
    """
    assignments = []
    for person_i, person_box in enumerate(people_xyxy):
        for face_i, face_box in enumerate(faces_xyxy):
            face_center = [(face_box[0] + face_box[2]) / 2, (face_box[1] + face_box[3]) / 2]
            if person_box[0] <= face_center[0] <= person_box[2] and person_box[1] <= face_center[1] <= person_box[3]:
                assignments.append((person_i, face_i))
                break

    print(f"Assigned {len(assignments)} faces to people")

    return assignments


@task(log_prints=True, retries=3, retry_delay_seconds=exponential_backoff(backoff_factor=3))
def store_image_results(image_url, people_xyxy, faces_xyxy, faces_people_assignment, flow_run_id):
    """
    Store the results to mongoDB
    """
    client = pymongo.MongoClient(MONGODB_CONNECTION_STRING)
    db = client["images"]
    collection = db["images"]

    data = {
        "url": image_url,
        "people_xyxy": people_xyxy,
        "faces_xyxy": faces_xyxy,
        "faces_people_assignment": faces_people_assignment,
        "flow_run_id": str(flow_run_id)
    }

    collection.insert_one(data)

    print(f"Stored results for {image_url}")


@flow(log_prints=True)
def url_faces_people_detection_flow(url):
    """
    Flow to process images from a given URL
    """
    flow_run_id = uuid.uuid4()
    images_location = os.path.join(IMAGES_DIRECTORY, str(flow_run_id))
    if not os.path.exists(images_location):
        os.makedirs(images_location)

    print(f"Processing images from {url}, flow run ID: {flow_run_id}")

    images_in_url = list_images_in_url(url)

    for image_url in images_in_url:
        image_path = download_image.submit(image_url, images_location)
        converted_image_path = convert_image(image_path)
        people_xyxy = detect_people.submit(converted_image_path)
        faces_xyxy = detect_faces.submit(converted_image_path)
        faces_people_assignment = assign_faces_to_people.submit(people_xyxy, faces_xyxy)
        store_image_results.submit(image_url, people_xyxy, faces_xyxy, faces_people_assignment, flow_run_id)

    shutil.rmtree(images_location)
    print(f"Finished processing images from {url}, flow run ID: {flow_run_id}")


if __name__ == "__main__":
    url_faces_people_detection_flow("https://www.nytimes.com/international/")
