# Copyright (C) 2021 DB Systel GmbH.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from PIL import Image
import coremltools as ct
import os
import numpy as np
import cv2
import json
from json import JSONEncoder
from pathlib import Path

from objectDetectionMetrics.BoundingBox import BoundingBox
from objectDetectionMetrics.BoundingBoxes import BoundingBoxes
from objectDetectionMetrics.Evaluator import *
from objectDetectionMetrics.utils import *

from argparse import ArgumentParser


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


classLabels = [
    "Person",
    "Sneakers",
    "Chair",
    "Other Shoes",
    "Hat",
    "Car",
    "Lamp",
    "Glasses",
    "Bottle",
    "Desk",
    "Cup",
    "Street Lights",
    "Cabinet-shelf",
    "Handbag-Satchel",
    "Bracelet",
    "Plate",
    "Picture-Frame",
    "Helmet",
    "Book",
    "Gloves",
    "Storage box",
    "Boat",
    "Leather Shoes",
    "Flower",
    "Bench",
    "Potted Plant",
    "Bowl-Basin",
    "Flag",
    "Pillow",
    "Boots",
    "Vase",
    "Microphone",
    "Necklace",
    "Ring",
    "SUV",
    "Wine Glass",
    "Belt",
    "Monitor-TV",
    "Backpack",
    "Umbrella",
    "Traffic Light",
    "Speaker",
    "Watch",
    "Tie",
    "Trash bin Can",
    "Slippers",
    "Bicycle",
    "Stool",
    "Barrel-bucket",
    "Van",
    "Couch",
    "Sandals",
    "Basket",
    "Drum",
    "Pen-Pencil",
    "Bus",
    "Wild Bird",
    "High Heels",
    "Motorcycle",
    "Guitar",
    "Carpet",
    "Cell Phone",
    "Bread",
    "Camera",
    "Canned",
    "Truck",
    "Traffic cone",
    "Cymbal",
    "Lifesaver",
    "Towel",
    "Stuffed Toy",
    "Candle",
    "Sailboat",
    "Laptop",
    "Awning",
    "Bed",
    "Faucet",
    "Tent",
    "Horse",
    "Mirror",
    "Power outlet",
    "Sink",
    "Apple",
    "Air Conditioner",
    "Knife",
    "Hockey Stick",
    "Paddle",
    "Pickup Truck",
    "Fork",
    "Traffic Sign",
    "Balloon",
    "Tripod",
    "Dog",
    "Spoon",
    "Clock",
    "Pot",
    "Cow",
    "Cake",
    "Dinning Table",
    "Sheep",
    "Hanger",
    "Blackboard-Whiteboard",
    "Napkin",
    "Other Fish",
    "Orange-Tangerine",
    "Toiletry",
    "Keyboard",
    "Tomato",
    "Lantern",
    "Machinery Vehicle",
    "Fan",
    "Green Vegetables",
    "Banana",
    "Baseball Glove",
    "Airplane",
    "Mouse",
    "Train",
    "Pumpkin",
    "Soccer",
    "Skiboard",
    "Luggage",
    "Nightstand",
    "Tea pot",
    "Telephone",
    "Trolley",
    "Head Phone",
    "Sports Car",
    "Stop Sign",
    "Dessert",
    "Scooter",
    "Stroller",
    "Crane",
    "Remote",
    "Refrigerator",
    "Oven",
    "Lemon",
    "Duck",
    "Baseball Bat",
    "Surveillance Camera",
    "Cat",
    "Jug",
    "Broccoli",
    "Piano",
    "Pizza",
    "Elephant",
    "Skateboard",
    "Surfboard",
    "Gun",
    "Skating and Skiing shoes",
    "Gas stove",
    "Donut",
    "Bow Tie",
    "Carrot",
    "Toilet",
    "Kite",
    "Strawberry",
    "Other Balls",
    "Shovel",
    "Pepper",
    "Computer Box",
    "Toilet Paper",
    "Cleaning Products",
    "Chopsticks",
    "Microwave",
    "Pigeon",
    "Baseball",
    "Cutting-chopping Board",
    "Coffee Table",
    "Side Table",
    "Scissors",
    "Marker",
    "Pie",
    "Ladder",
    "Snowboard",
    "Cookies",
    "Radiator",
    "Fire Hydrant",
    "Basketball",
    "Zebra",
    "Grape",
    "Giraffe",
    "Potato",
    "Sausage",
    "Tricycle",
    "Violin",
    "Egg",
    "Fire Extinguisher",
    "Candy",
    "Fire Truck",
    "Billiards",
    "Converter",
    "Bathtub",
    "Wheelchair",
    "Golf Club",
    "Briefcase",
    "Cucumber",
    "Cigar-Cigarette",
    "Paint Brush",
    "Pear",
    "Heavy Truck",
    "Hamburger",
    "Extractor",
    "Extension Cord",
    "Tong",
    "Tennis Racket",
    "Folder",
    "American Football",
    "earphone",
    "Mask",
    "Kettle",
    "Tennis",
    "Ship",
    "Swing",
    "Coffee Machine",
    "Slide",
    "Carriage",
    "Onion",
    "Green beans",
    "Projector",
    "Frisbee",
    "Washing Machine-Drying Machine",
    "Chicken",
    "Printer",
    "Watermelon",
    "Saxophone",
    "Tissue",
    "Toothbrush",
    "Ice cream",
    "Hot-air balloon",
    "Cello",
    "French Fries",
    "Scale",
    "Trophy",
    "Cabbage",
    "Hot dog",
    "Blender",
    "Peach",
    "Rice",
    "Wallet-Purse",
    "Volleyball",
    "Deer",
    "Goose",
    "Tape",
    "Tablet",
    "Cosmetics",
    "Trumpet",
    "Pineapple",
    "Golf Ball",
    "Ambulance",
    "Parking meter",
    "Mango",
    "Key",
    "Hurdle",
    "Fishing Rod",
    "Medal",
    "Flute",
    "Brush",
    "Penguin",
    "Megaphone",
    "Corn",
    "Lettuce",
    "Garlic",
    "Swan",
    "Helicopter",
    "Green Onion",
    "Sandwich",
    "Nuts",
    "Speed Limit Sign",
    "Induction Cooker",
    "Broom",
    "Trombone",
    "Plum",
    "Rickshaw",
    "Goldfish",
    "Kiwi fruit",
    "Router-modem",
    "Poker Card",
    "Toaster",
    "Shrimp",
    "Sushi",
    "Cheese",
    "Notepaper",
    "Cherry",
    "Pliers",
    "CD",
    "Pasta",
    "Hammer",
    "Cue",
    "Avocado",
    "Hamimelon",
    "Flask",
    "Mushroom",
    "Screwdriver",
    "Soap",
    "Recorder",
    "Bear",
    "Eggplant",
    "Board Eraser",
    "Coconut",
    "Tape Measure-Ruler",
    "Pig",
    "Showerhead",
    "Globe",
    "Chips",
    "Steak",
    "Crosswalk Sign",
    "Stapler",
    "Camel",
    "Formula 1",
    "Pomegranate",
    "Dishwasher",
    "Crab",
    "Hoverboard",
    "Meat ball",
    "Rice Cooker",
    "Tuba",
    "Calculator",
    "Papaya",
    "Antelope",
    "Parrot",
    "Seal",
    "Butterfly",
    "Dumbbell",
    "Donkey",
    "Lion",
    "Urinal",
    "Dolphin",
    "Electric Drill",
    "Hair Dryer",
    "Egg tart",
    "Jellyfish",
    "Treadmill",
    "Lighter",
    "Grapefruit",
    "Game board",
    "Mop",
    "Radish",
    "Baozi",
    "Target",
    "French",
    "Spring Rolls",
    "Monkey",
    "Rabbit",
    "Pencil Case",
    "Yak",
    "Red Cabbage",
    "Binoculars",
    "Asparagus",
    "Barbell",
    "Scallop",
    "Noddles",
    "Comb",
    "Dumpling",
    "Oyster",
    "Table Tennis paddle",
    "Cosmetics Brush-Eyeliner Pencil",
    "Chainsaw",
    "Eraser",
    "Lobster",
    "Durian",
    "Okra",
    "Lipstick",
    "Cosmetics Mirror",
    "Curling",
    "Table Tennis",
]
LABELS = {}
for i, name in enumerate(classLabels):
    LABELS[str(i)] = name

IMAGE_FILE_SUFFIXES = (".jpeg", ".jpg")
MAX_IMAGES = 100
# Model config
IOU_THRESHOLD = 0.3
IOU_THRESHOLD_MODEL = 0.3
CONFIDENCE_THRESHOLD = 0.3

IMAGE_SIZE = 640


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--model-input-path",
        type=str,
        dest="model_input_path",
        default="output/models/yolov5-iOS.mlmodel",
        help="path to coreml model",
    )
    parser.add_argument(
        "--image-folder",
        type=str,
        dest="image_folder",
        default="data/images",
        help="path to image root folder",
    )
    parser.add_argument(
        "--label-folder",
        type=str,
        dest="label_folder",
        default="data/labels",
        help="path to label root folder (folder needs to mirror directory structure of the image folder)",
    )
    parser.add_argument(
        "--metrics_output-directory",
        type=str,
        dest="metrics_output_directory",
        default="output/metrics",
        help="path to metrics output folder (will be created if it does not exist)",
    )
    opt = parser.parse_args()

    Path(opt.metrics_output_directory).mkdir(parents=True, exist_ok=True)

    allBoundingBoxes = queryFolders(
        opt.model_input_path,
        opt.image_folder,
        opt.label_folder,
        opt.metrics_output_directory,
    )


# Will scan all subdirectories recursively and write an evulation for all images found in a directory and its child directory
def queryFolders(model, imageFolder, labelsFolder, outputDirectory):
    allBoundingBoxes = BoundingBoxes()
    for subImageFolder in os.scandir(imageFolder):
        # Recursive call for subfolders
        if subImageFolder.is_dir():
            subBoundingBoxes = queryFolders(
                model,
                subImageFolder.path,
                f"{labelsFolder}/{subImageFolder.name}",
                outputDirectory,
            )
            allBoundingBoxes.addBoundingBoxes(subBoundingBoxes)

    boundingBoxes = analyseCurrentDir(model, imageFolder, labelsFolder, outputDirectory)
    if boundingBoxes:
        allBoundingBoxes.addBoundingBoxes(boundingBoxes)

    # Check if directory and subdirectories contain any image at all
    if not allBoundingBoxes:
        return

    metricsOutputFolder = imageFolder.replace("data", outputDirectory) + "/metrics"
    Path(metricsOutputFolder).mkdir(parents=True, exist_ok=True)

    print(f"Evaluate {imageFolder}")
    evaluate(allBoundingBoxes, metricsOutputFolder)
    return allBoundingBoxes


def analyseCurrentDir(model, imageFolder, labelsFolder, outputDirectory):
    imageEntries = [
        imageEntry
        for imageEntry in os.scandir(imageFolder)
        if imageEntry.name.endswith(IMAGE_FILE_SUFFIXES)
    ]

    if not imageEntries:
        return

    if not Path(labelsFolder):
        print(f"Labels folder {labelsFolder} for {imageFolder} doesn't exist")
        return

    detectionOutputFolder = imageFolder.replace("data", outputDirectory) + "/detections"
    Path(detectionOutputFolder).mkdir(parents=True, exist_ok=True)

    imageOutputFolder = imageFolder.replace("data", outputDirectory) + "/images"
    Path(imageOutputFolder).mkdir(parents=True, exist_ok=True)

    detectCoreML(model, imageEntries, detectionOutputFolder)
    boundingBoxes = getBoundingBoxes(labelsFolder, detectionOutputFolder)
    drawBoundingBox(imageEntries, boundingBoxes, imageOutputFolder)

    return boundingBoxes


def evaluate(boundingBoxes, metricsOutputFolder):
    try:
        metricsList = Evaluator().PlotPrecisionRecallCurve(
            boundingBoxes,
            IOUThreshold=IOU_THRESHOLD,
            method=MethodAveragePrecision.EveryPointInterpolation,
            showAP=True,
            showInterpolatedPrecision=True,
            savePath=metricsOutputFolder,
            showGraphic=False,
        )
        with open(f"{metricsOutputFolder}/metrics.json", "w") as metricsFile:
            json.dump(metricsList, metricsFile, cls=NumpyArrayEncoder)
    except Exception as e:
        print(e)


def detectCoreML(modelPath, imageEntries, outputFolder):
    model = ct.models.MLModel(modelPath, useCPUOnly=True)
    maxProcess = MAX_IMAGES
    for imageEntry in imageEntries:
        maxProcess -= 1
        if maxProcess <= 0:
            break
        inputImage = Image.open(imageEntry.path).resize((640, 640))

        out_dict = model.predict(
            {
                "image": inputImage,
                "iouThreshold": IOU_THRESHOLD_MODEL,
                "confidenceThreshold": CONFIDENCE_THRESHOLD,
            }
        )

        outFileName = Path(imageEntry.path).stem
        outFilePath = f"{outputFolder}/{outFileName}.txt"

        with open(outFilePath, "w") as outFile:
            for coordinates, confidence in zip(
                out_dict["coordinates"], out_dict["confidence"]
            ):
                labelMax = confidence.argmax()
                outFile.write(
                    "{:d} {:.6f} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(
                        labelMax,
                        coordinates[0],
                        coordinates[1],
                        coordinates[2],
                        coordinates[3],
                        confidence[labelMax],
                    )
                )

        print(f"Image {outFileName} predicted!")


# Convert validation and detection YOLO files into Bounding Box Objects
def getBoundingBoxes(valFolder, detectFolder):
    boundingBoxes = BoundingBoxes()
    addBoundingBoxes(boundingBoxes, valFolder, isGroundTruth=True)
    addBoundingBoxes(boundingBoxes, detectFolder, isGroundTruth=False)
    return boundingBoxes


# Convert label YOLO files into Bounding Box Objects
def addBoundingBoxes(boundingBoxes, labelFolder, isGroundTruth):
    for labelFileEntry in os.scandir(labelFolder):
        if not labelFileEntry.name.endswith(".txt"):
            continue

        if not Path(labelFileEntry.path):
            print(f"Missing label file {labelFileEntry.path}")
            continue

        imageName = Path(labelFileEntry.path).stem
        with open(labelFileEntry.path, "r") as labelFile:
            for labelLine in labelFile:
                labelNumbers = labelLine.split()

                # ignore empty lines
                if len(labelNumbers) == 0:
                    continue

                if len(labelNumbers) < 5:
                    print(
                        f"Warning: Not enough values in some line in {groundTruthFolder}/{groundTruthFileName}"
                    )
                    continue

                if isGroundTruth:
                    bb = BoundingBox(
                        imageName,
                        LABELS[labelNumbers[0]],
                        float(labelNumbers[1]),
                        float(labelNumbers[2]),
                        float(labelNumbers[3]),
                        float(labelNumbers[4]),
                        CoordinatesType.Relative,
                        (IMAGE_SIZE, IMAGE_SIZE),
                        BBType.GroundTruth,
                        format=BBFormat.XYWH,
                    )
                else:
                    bb = BoundingBox(
                        imageName,
                        LABELS[labelNumbers[0]],
                        float(labelNumbers[1]),
                        float(labelNumbers[2]),
                        float(labelNumbers[3]),
                        float(labelNumbers[4]),
                        CoordinatesType.Relative,
                        (IMAGE_SIZE, IMAGE_SIZE),
                        BBType.Detected,
                        float(labelNumbers[5]),
                        format=BBFormat.XYWH,
                    )
                boundingBoxes.addBoundingBox(bb)


def drawBoundingBox(imageEntries, boundingBoxes, outputFolder):
    maxProcess = MAX_IMAGES
    for imageEntry in imageEntries:
        maxProcess -= 1
        if maxProcess <= 0:
            break
        imageName = Path(imageEntry.path).stem

        # Read image and resize to model image size
        image = cv2.imread(imageEntry.path)
        (originalHeight, originalWidth) = image.shape[:2]
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        image = boundingBoxes.drawAllBoundingBoxes(image, imageName)

        # Resize image back to original site and write to file
        image = cv2.resize(image, (originalWidth, originalHeight))
        cv2.imwrite(f"{outputFolder}/{imageEntry.name}", image)

        print(f"Image {imageName} boundingBoxes created successfully!")
