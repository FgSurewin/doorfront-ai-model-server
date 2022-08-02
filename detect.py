import time
import torch
from torchvision import transforms
import requests
from PIL import Image
from io import BytesIO
import os
import warnings
import sys
import torchvision
import pprint

pp = pprint.PrettyPrinter(indent=4)


warnings.filterwarnings("ignore")


def load_model(path):
    device = torch.device("cpu")
    model = torch.load(path)
    model.to(device)
    model.eval()  # evaluation
    return model


def load_path(file):
    current_file = os.path.abspath(__file__)
    _dirname = os.path.dirname(current_file)
    return os.path.abspath(_dirname + os.path.sep + file)


def get_result(nms_threshold, prediction):
    predict_boxes_list = prediction["boxes"].to("cpu").numpy()
    predict_classes_list = prediction["labels"].to("cpu").numpy()
    predict_scores_list = prediction["scores"].to("cpu").numpy()

    # Apply NMS
    tensor_boxes = prediction["boxes"].to("cpu")
    tensor_scores = prediction["scores"].to("cpu")
    keep_index = torchvision.ops.nms(tensor_boxes, tensor_scores, nms_threshold)

    # predict_boxes = {"predict_boxes": predict_boxes_list[(keep_index.numpy().tolist())].tolist()}
    # predict_classes = {"predict_classes": predict_classes_list[(keep_index.numpy().tolist())].tolist()}
    # predict_scores = {"predict_scores":predict_scores_list[(keep_index.numpy().tolist())].tolist()}

    return zip(
        predict_boxes_list[(keep_index.numpy().tolist())].tolist(),
        predict_classes_list[(keep_index.numpy().tolist())].tolist(),
        predict_scores_list[(keep_index.numpy().tolist())].tolist(),
    )


def detect(nms_threshold, url):
    response = requests.get(url)
    img = Image.open(BytesIO(response.content))
    # img = Image.open(load_path("test.jpg"))
    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0] and convert [H,W,C] to [C,H,W]
        ]
    )
    img = img_transform(img)
    img = torch.unsqueeze(img, dim=0)
    # print(img)

    # # read class_indict
    # label_json_path = './storefront_classes.json'
    # assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    # json_file = open(label_json_path, 'r')
    # class_dict = json.load(json_file)
    # category_index = {v: k for k, v in class_dict.items()}

    model = load_model(load_path("model.pkl"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device: ", device)
    with torch.no_grad():
        model.to(device)
        predict_result = model(img.to(device))

        predictions = predict_result[0]

        return get_result(nms_threshold, predictions)
        # predict_boxes_list = predictions["boxes"].to("cpu").numpy()
        # predict_classes_list = predictions["labels"].to("cpu").numpy()
        # predict_scores_list = predictions["scores"].to("cpu").numpy()

        # # Apply NMS
        # tensor_boxes = predictions["boxes"].to("cpu")
        # tensor_scores = predictions["scores"].to("cpu")
        # keep_index = torchvision.ops.nms(tensor_boxes, tensor_scores, nms_threshold)

        # return zip(
        #     predict_boxes_list[(keep_index.numpy().tolist())].tolist(),
        #     predict_classes_list[(keep_index.numpy().tolist())].tolist(),
        #     predict_scores_list[(keep_index.numpy().tolist())].tolist(),
        # )


def detect_multiple_images(nms_threshold, image_list):
    st = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    img_transform = transforms.Compose(
        [
            transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0] and convert [H,W,C] to [C,H,W]
        ]
    )
    ### --------------------------------------------------
    image_id_list = [image["imageId"] for image in image_list]
    image_list = [
        img_transform(Image.open(BytesIO(requests.get(image["imgSrc"]).content)))
        for image in image_list
    ]
    image_list = torch.stack(image_list)
    print(image_list.shape)
    model = load_model(load_path("model.pkl"))
    with torch.no_grad():
        model.to(device)
        predict_result = model(image_list.to(device))
        et = time.time()
        duration = et - st
        print("duration: ", duration)
        final_result = [
            {
                "image_id": image_id,
                "labels": list(get_result(nms_threshold, prediction)),
            }
            for image_id, prediction in zip(image_id_list, predict_result)
        ]
        return final_result


# if __name__ == "__main__":
#     # URL = sys.argv[1]
#     URL = "https://firebasestorage.googleapis.com/v0/b/crsp--streetview.appspot.com/o/FkGKKUcKilWN4BIpNNHMrQ%2Fh_30.161314202981345_p_1.2450773755023334?alt=media&token=a30dbd6e-a252-4636-880d-88a07ff82e25"
#     detect(0.2, URL)
