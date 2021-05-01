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

import torch
import torch.nn as nn
import coremltools as ct
from argparse import ArgumentParser
from pathlib import Path

# Add silu function for yolov5s v4 model: https://github.com/apple/coremltools/issues/1099
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs


@register_torch_op
def silu(context, node):
    inputs = _get_inputs(context, node, expected=1)
    x = inputs[0]
    y = mb.sigmoid(x=x)
    z = mb.mul(x=x, y=y, name=node.name)
    context.add(z)


# The labels of your model, pretrained YOLOv5 models usually use the coco dataset and have 80 classes
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
    "Cabinet/shelf",
    "Handbag/Satchel",
    "Bracelet",
    "Plate",
    "Picture/Frame",
    "Helmet",
    "Book",
    "Gloves",
    "Storage box",
    "Boat",
    "Leather Shoes",
    "Flower",
    "Bench",
    "Potted Plant",
    "Bowl/Basin",
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
    "Monitor/TV",
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
    "Barrel/bucket",
    "Van",
    "Couch",
    "Sandals",
    "Basket",
    "Drum",
    "Pen/Pencil",
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
    "Blackboard/Whiteboard",
    "Napkin",
    "Other Fish",
    "Orange/Tangerine",
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
    "Cutting/chopping Board",
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
    "Cigar/Cigarette",
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
    "Washing Machine/Drying Machine",
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
    "Wallet/Purse",
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
    "Router/modem",
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
    "Tape Measure/Ruler",
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
    "Cosmetics Brush/Eyeliner Pencil",
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
numberOfClassLabels = len(classLabels)
outputSize = numberOfClassLabels + 5

#  Attention: Some models are reversed!
reverseModel = False

strides = [8, 16, 32]
if reverseModel:
    strides.reverse()
featureMapDimensions = [640 // stride for stride in strides]

anchors = (
    [10, 13, 16, 30, 33, 23],
    [30, 61, 62, 45, 59, 119],
    [116, 90, 156, 198, 373, 326],
)  # Take these from the <model>.yml in yolov5
if reverseModel:
    anchors = anchors[::-1]

anchorGrid = torch.tensor(anchors).float().view(3, -1, 1, 1, 2)


def make_grid(nx, ny):
    yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
    return torch.stack((xv, yv), 2).view((ny, nx, 2)).float()


def exportTorchscript(model, sampleInput, checkInputs, fileName):
    """
    Traces a pytorch model and produces a TorchScript
    """
    try:
        print(f"Starting TorchScript export with torch {torch.__version__}")
        ts = torch.jit.trace(model, sampleInput, check_inputs=checkInputs)
        ts.save(fileName)
        print(f"TorchScript export success, saved as {fileName}")
        return ts
    except Exception as e:
        print(f"TorchScript export failure: {e}")


def convertToCoremlSpec(torchScript, sampleInput):
    """
    Converts a torchscript to a coreml model
    """
    try:
        print(f"Starting CoreML conversion with coremltools {ct.__version__}")
        nnSpec = ct.convert(
            torchScript,
            inputs=[
                ct.ImageType(
                    name="image",
                    shape=sampleInput.shape,
                    scale=1 / 255.0,
                    bias=[0, 0, 0],
                )
            ],
        ).get_spec()

        print(f"CoreML conversion success")
    except Exception as e:
        print(f"CoreML conversion failure: {e}")
        return
    return nnSpec


def addOutputMetaData(nnSpec):
    """
    Adds the correct output shapes and data types to the coreml model
    """
    for i, featureMapDimension in enumerate(featureMapDimensions):
        nnSpec.description.output[i].type.multiArrayType.shape.append(1)
        nnSpec.description.output[i].type.multiArrayType.shape.append(3)
        nnSpec.description.output[i].type.multiArrayType.shape.append(
            featureMapDimension
        )
        nnSpec.description.output[i].type.multiArrayType.shape.append(
            featureMapDimension
        )
        # pc, bx, by, bh, bw, c (no of class class labels)
        nnSpec.description.output[i].type.multiArrayType.shape.append(outputSize)
        nnSpec.description.output[
            i
        ].type.multiArrayType.dataType = (
            ct.proto.FeatureTypes_pb2.ArrayFeatureType.DOUBLE
        )


def addExportLayerToCoreml(builder):
    """
    Adds the yolov5 export layer to the coreml model
    """
    outputNames = [output.name for output in builder.spec.description.output]

    for i, outputName in enumerate(outputNames):
        # formulas: https://github.com/ultralytics/yolov5/issues/471
        builder.add_activation(
            name=f"sigmoid_{outputName}",
            non_linearity="SIGMOID",
            input_name=outputName,
            output_name=f"{outputName}_sigmoid",
        )

        ### Coordinates calculation ###
        # input (1, 3, nC, nC, 85), output (1, 3, nC, nC, 2) -> nC = 640 / strides[i]
        builder.add_slice(
            name=f"slice_coordinates_xy_{outputName}",
            input_name=f"{outputName}_sigmoid",
            output_name=f"{outputName}_sliced_coordinates_xy",
            axis="width",
            start_index=0,
            end_index=2,
        )
        # x,y * 2
        builder.add_elementwise(
            name=f"multiply_xy_by_two_{outputName}",
            input_names=[f"{outputName}_sliced_coordinates_xy"],
            output_name=f"{outputName}_multiplied_xy_by_two",
            mode="MULTIPLY",
            alpha=2,
        )
        # x,y * 2 - 0.5
        builder.add_elementwise(
            name=f"subtract_0_5_from_xy_{outputName}",
            input_names=[f"{outputName}_multiplied_xy_by_two"],
            output_name=f"{outputName}_subtracted_0_5_from_xy",
            mode="ADD",
            alpha=-0.5,
        )
        grid = make_grid(featureMapDimensions[i], featureMapDimensions[i]).numpy()
        # x,y * 2 - 0.5 + grid[i]
        builder.add_bias(
            name=f"add_grid_from_xy_{outputName}",
            input_name=f"{outputName}_subtracted_0_5_from_xy",
            output_name=f"{outputName}_added_grid_xy",
            b=grid,
            shape_bias=grid.shape,
        )
        # (x,y * 2 - 0.5 + grid[i]) * stride[i]
        builder.add_elementwise(
            name=f"multiply_xy_by_stride_{outputName}",
            input_names=[f"{outputName}_added_grid_xy"],
            output_name=f"{outputName}_calculated_xy",
            mode="MULTIPLY",
            alpha=strides[i],
        )

        # input (1, 3, nC, nC, 85), output (1, 3, nC, nC, 2)
        builder.add_slice(
            name=f"slice_coordinates_wh_{outputName}",
            input_name=f"{outputName}_sigmoid",
            output_name=f"{outputName}_sliced_coordinates_wh",
            axis="width",
            start_index=2,
            end_index=4,
        )
        # w,h * 2
        builder.add_elementwise(
            name=f"multiply_wh_by_two_{outputName}",
            input_names=[f"{outputName}_sliced_coordinates_wh"],
            output_name=f"{outputName}_multiplied_wh_by_two",
            mode="MULTIPLY",
            alpha=2,
        )
        # (w,h * 2) ** 2
        builder.add_unary(
            name=f"power_wh_{outputName}",
            input_name=f"{outputName}_multiplied_wh_by_two",
            output_name=f"{outputName}_power_wh",
            mode="power",
            alpha=2,
        )
        # (w,h * 2) ** 2 * anchor_grid[i]
        anchor = (
            anchorGrid[i]
            .expand(-1, featureMapDimensions[i], featureMapDimensions[i], -1)
            .numpy()
        )
        builder.add_load_constant_nd(
            name=f"anchors_{outputName}",
            output_name=f"{outputName}_anchors",
            constant_value=anchor,
            shape=anchor.shape,
        )
        builder.add_elementwise(
            name=f"multiply_wh_with_achors_{outputName}",
            input_names=[f"{outputName}_power_wh", f"{outputName}_anchors"],
            output_name=f"{outputName}_calculated_wh",
            mode="MULTIPLY",
        )

        builder.add_concat_nd(
            name=f"concat_coordinates_{outputName}",
            input_names=[f"{outputName}_calculated_xy", f"{outputName}_calculated_wh"],
            output_name=f"{outputName}_raw_coordinates",
            axis=-1,
        )
        builder.add_scale(
            name=f"normalize_coordinates_{outputName}",
            input_name=f"{outputName}_raw_coordinates",
            output_name=f"{outputName}_raw_normalized_coordinates",
            W=torch.tensor([1 / 640]).numpy(),
            b=0,
            has_bias=False,
        )

        ### Confidence calculation ###
        builder.add_slice(
            name=f"slice_object_confidence_{outputName}",
            input_name=f"{outputName}_sigmoid",
            output_name=f"{outputName}_object_confidence",
            axis="width",
            start_index=4,
            end_index=5,
        )
        builder.add_slice(
            name=f"slice_label_confidence_{outputName}",
            input_name=f"{outputName}_sigmoid",
            output_name=f"{outputName}_label_confidence",
            axis="width",
            start_index=5,
            end_index=0,
        )
        # confidence = object_confidence * label_confidence
        builder.add_multiply_broadcastable(
            name=f"multiply_object_label_confidence_{outputName}",
            input_names=[
                f"{outputName}_label_confidence",
                f"{outputName}_object_confidence",
            ],
            output_name=f"{outputName}_raw_confidence",
        )

        # input: (1, 3, nC, nC, 85), output: (3 * nc^2, 85)
        builder.add_flatten_to_2d(
            name=f"flatten_confidence_{outputName}",
            input_name=f"{outputName}_raw_confidence",
            output_name=f"{outputName}_flatten_raw_confidence",
            axis=-1,
        )
        builder.add_flatten_to_2d(
            name=f"flatten_coordinates_{outputName}",
            input_name=f"{outputName}_raw_normalized_coordinates",
            output_name=f"{outputName}_flatten_raw_coordinates",
            axis=-1,
        )

    builder.add_concat_nd(
        name="concat_confidence",
        input_names=[
            f"{outputName}_flatten_raw_confidence" for outputName in outputNames
        ],
        output_name="raw_confidence",
        axis=-2,
    )
    builder.add_concat_nd(
        name="concat_coordinates",
        input_names=[
            f"{outputName}_flatten_raw_coordinates" for outputName in outputNames
        ],
        output_name="raw_coordinates",
        axis=-2,
    )

    builder.set_output(
        output_names=["raw_confidence", "raw_coordinates"],
        output_dims=[(25200, numberOfClassLabels), (25200, 4)],
    )


def createNmsModelSpec(nnSpec):
    """
    Create a coreml model with nms to filter the results of the model
    """
    nmsSpec = ct.proto.Model_pb2.Model()
    nmsSpec.specificationVersion = 4

    # Define input and outputs of the model
    for i in range(2):
        nnOutput = nnSpec.description.output[i].SerializeToString()

        nmsSpec.description.input.add()
        nmsSpec.description.input[i].ParseFromString(nnOutput)

        nmsSpec.description.output.add()
        nmsSpec.description.output[i].ParseFromString(nnOutput)

    nmsSpec.description.output[0].name = "confidence"
    nmsSpec.description.output[1].name = "coordinates"

    # Define output shape of the model
    outputSizes = [numberOfClassLabels, 4]
    for i in range(len(outputSizes)):
        maType = nmsSpec.description.output[i].type.multiArrayType
        # First dimension of both output is the number of boxes, which should be flexible
        maType.shapeRange.sizeRanges.add()
        maType.shapeRange.sizeRanges[0].lowerBound = 0
        maType.shapeRange.sizeRanges[0].upperBound = -1
        # Second dimension is fixed, for "confidence" it's the number of classes, for coordinates it's position (x, y) and size (w, h)
        maType.shapeRange.sizeRanges.add()
        maType.shapeRange.sizeRanges[1].lowerBound = outputSizes[i]
        maType.shapeRange.sizeRanges[1].upperBound = outputSizes[i]
        del maType.shape[:]

    # Define the model type non maximum supression
    nms = nmsSpec.nonMaximumSuppression
    nms.confidenceInputFeatureName = "raw_confidence"
    nms.coordinatesInputFeatureName = "raw_coordinates"
    nms.confidenceOutputFeatureName = "confidence"
    nms.coordinatesOutputFeatureName = "coordinates"
    nms.iouThresholdInputFeatureName = "iouThreshold"
    nms.confidenceThresholdInputFeatureName = "confidenceThreshold"
    # Some good default values for the two additional inputs, can be overwritten when using the model
    nms.iouThreshold = 0.6
    nms.confidenceThreshold = 0.25
    nms.stringClassLabels.vector.extend(classLabels)

    return nmsSpec


def combineModelsAndExport(builderSpec, nmsSpec, fileName, quantize=False):
    """
    Combines the coreml model with export logic and the nms to one final model. Optionally save with different quantization (32, 16, 8) (Works only if on Mac Os)
    """
    try:
        print(f"Combine CoreMl model with nms and export model")
        # Combine models to a single one
        pipeline = ct.models.pipeline.Pipeline(
            input_features=[
                ("image", ct.models.datatypes.Array(3, 460, 460)),
                ("iouThreshold", ct.models.datatypes.Double()),
                ("confidenceThreshold", ct.models.datatypes.Double()),
            ],
            output_features=["confidence", "coordinates"],
        )

        # Required version (>= ios13) in order for mns to work
        pipeline.spec.specificationVersion = 4

        pipeline.add_model(builderSpec)
        pipeline.add_model(nmsSpec)

        pipeline.spec.description.input[0].ParseFromString(
            builderSpec.description.input[0].SerializeToString()
        )
        pipeline.spec.description.output[0].ParseFromString(
            nmsSpec.description.output[0].SerializeToString()
        )
        pipeline.spec.description.output[1].ParseFromString(
            nmsSpec.description.output[1].SerializeToString()
        )

        # Metadata for the model‚
        pipeline.spec.description.input[
            1
        ].shortDescription = "(optional) IOU Threshold override (Default: 0.6)"
        pipeline.spec.description.input[
            2
        ].shortDescription = "(optional) Confidence Threshold override (Default: 0.4)"
        pipeline.spec.description.output[
            0
        ].shortDescription = "Boxes \xd7 Class confidence"
        pipeline.spec.description.output[
            1
        ].shortDescription = "Boxes \xd7 [x, y, width, height] (relative to image size)"
        pipeline.spec.description.metadata.versionString = "1.0"
        pipeline.spec.description.metadata.shortDescription = "yolov5"
        pipeline.spec.description.metadata.author = "Leon De Andrade"
        pipeline.spec.description.metadata.license = ""

        model = ct.models.MLModel(pipeline.spec)
        model.save(fileName)

        if quantize:
            fileName16 = fileName.replace(".mlmodel", "_16.mlmodel")
            modelFp16 = ct.models.neural_network.quantization_utils.quantize_weights(
                model, nbits=16
            )
            modelFp16.save(fileName16)

            fileName8 = fileName.replace(".mlmodel", "_8.mlmodel")
            modelFp8 = ct.models.neural_network.quantization_utils.quantize_weights(
                model, nbits=8
            )
            modelFp8.save(fileName8)

        print(f"CoreML export success, saved as {fileName}")
    except Exception as e:
        print(f"CoreML export failure: {e}")


class BatchNormXd(torch.nn.modules.batchnorm._BatchNorm):
    def _check_input_dim(self, input):
        # The only difference between BatchNorm1d, BatchNorm2d, BatchNorm3d, etc
        # is this method that is overwritten by the sub-class
        # This original goal of this method was for tensor sanity checks
        # If you're ok bypassing those sanity checks (eg. if you trust your inference
        # to provide the right dimensional inputs), then you can just use this method
        # for easy conversion from SyncBatchNorm
        # (unfortunately, SyncBatchNorm does not store the original class - if it did
        #  we could return the one that was originally created)
        return


def revert_sync_batchnorm(module):
    # this is very similar to the function that it is trying to revert:
    # https://github.com/pytorch/pytorch/blob/c8b3686a3e4ba63dc59e5dcfe5db3430df256833/torch/nn/modules/batchnorm.py#L679
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm.SyncBatchNorm):
        new_cls = BatchNormXd
        module_output = nn.BatchNorm2d(
            module.num_features,
            module.eps,
            module.momentum,
            module.affine,
            module.track_running_stats,
        )
        if module.affine:
            with torch.no_grad():
                module_output.weight = module.weight
                module_output.bias = module.bias
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
        module_output.num_batches_tracked = module.num_batches_tracked
        if hasattr(module, "qconfig"):
            module_output.qconfig = module.qconfig
    for name, child in module.named_children():
        module_output.add_module(name, revert_sync_batchnorm(child))
    del module
    return module_output


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            nn.SiLU()
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


def main():

    parser = ArgumentParser()
    parser.add_argument(
        "--model-input-path",
        type=str,
        dest="model_input_path",
        default="models/yolov5s_v4.pt",
        help="path to yolov5 model",
    )
    parser.add_argument(
        "--model-output-directory",
        type=str,
        dest="model_output_directory",
        default="output/models",
        help="model output path",
    )
    parser.add_argument(
        "--model-output-name",
        type=str,
        dest="model_output_name",
        default="yolov5-iOS",
        help="model output name",
    )
    parser.add_argument(
        "--quantize-model",
        action="store_true",
        dest="quantize",
        help="Pass flag quantized models are needed (Only works on mac Os)",
    )
    opt = parser.parse_args()

    if not Path(opt.model_input_path).exists():
        print("Error: Input model not found")
        return

    Path(opt.model_output_directory).mkdir(parents=True, exist_ok=True)

    sampleInput = torch.zeros((1, 3, 640, 640))
    checkInputs = [(torch.rand(1, 3, 640, 640),), (torch.rand(1, 3, 640, 640),)]

    model = torch.load(opt.model_input_path, map_location=torch.device("cpu"))[
        "model"
    ].float()

    model = revert_sync_batchnorm(model)
    print(model)

    # Update model
    for k, m in model.named_modules():
        m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    model.eval()
    model.model[-1].export = True
    # Dry run, necessary for correct tracing!
    model(sampleInput)

    ts = exportTorchscript(
        model,
        sampleInput,
        checkInputs,
        f"{opt.model_output_directory}/{opt.model_output_name}.torchscript.pt",
    )

    # Convert pytorch to raw coreml model
    modelSpec = convertToCoremlSpec(ts, sampleInput)
    addOutputMetaData(modelSpec)

    # Add export logic to coreml model
    builder = ct.models.neural_network.NeuralNetworkBuilder(spec=modelSpec)
    addExportLayerToCoreml(builder)

    # Create nms logic
    nmsSpec = createNmsModelSpec(builder.spec)

    # Combine model with export logic and nms logic
    combineModelsAndExport(
        builder.spec,
        nmsSpec,
        f"{opt.model_output_directory}/{opt.model_output_name}.mlmodel",
        opt.quantize,
    )


if __name__ == "__main__":
    main()
