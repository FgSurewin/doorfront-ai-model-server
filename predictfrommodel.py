import torch
import os
import json
import matplotlib.pyplot as plt
import collections
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
import numpy as np
from pathlib import Path
import os
import torchvision

from PIL import Image
from torchvision import transforms

STANDARD_COLORS = [
    'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
    'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
    'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
    'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
    'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
    'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
    'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
    'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
    'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
    'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
    'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
    'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
    'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
    'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
    'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
    'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
    'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
    'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
    'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
    'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
    'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
    'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
    'WhiteSmoke', 'Yellow', 'YellowGreen'
]

def load_path(file):
    current_file = os.path.abspath(__file__)
    _dirname = os.path.dirname(current_file)
    return (os.path.abspath(_dirname + os.path.sep+file))

def draw_text(draw, box_to_display_str_map, box, left, right, top, bottom, color):
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except IOError:
        font = ImageFont.load_default()

    # If the total height of the display strings added to the top of the bounding
    # box exceeds the top of the image, stack the strings below the bounding box
    # instead of above.
    display_str_heights = [font.getsize(ds)[1] for ds in box_to_display_str_map[box]]
    # Each display_str has a top and bottom margin of 0.05x.
    total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

    if top > total_display_str_height:
        text_bottom = top
    else:
        text_bottom = bottom + total_display_str_height
    # Reverse list and print from bottom to top.
    for display_str in box_to_display_str_map[box][::-1]:
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle([(left, text_bottom - text_height - 2 * margin),
                        (left + text_width, text_bottom)], fill=color)
        draw.text((left + margin, text_bottom - text_height - margin),
                  display_str,
                  fill='black',
                  font=font)
        text_bottom -= text_height - 2 * margin


def filter_low_thresh(boxes, scores, classes, category_index, thresh, box_to_display_str_map, box_to_color_map):
    for i in range(boxes.shape[0]):
        if scores[i] > thresh:
            box = tuple(boxes[i].tolist())  # numpy -> list -> tuple
            if classes[i] in category_index.keys():
                class_name = category_index[classes[i]]
            else:
                class_name = 'N/A'
            display_str = str(class_name)
            display_str = '{}: {}%'.format(display_str, int(100 * scores[i]))
            box_to_display_str_map[box].append(display_str)
            box_to_color_map[box] = STANDARD_COLORS[
                classes[i] % len(STANDARD_COLORS)]
        else:
            break


def draw_box(image, boxes, classes, scores, category_index, thresh=0.5, line_thickness=8):
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)

    filter_low_thresh(boxes, scores, classes, category_index, thresh, box_to_display_str_map, box_to_color_map)

    # Draw all boxes onto image.
    draw = ImageDraw.Draw(image)
    im_width, im_height = image.size
    for box, color in box_to_color_map.items():
        xmin, ymin, xmax, ymax = box
        (left, right, top, bottom) = (xmin * 1, xmax * 1,
                                      ymin * 1, ymax * 1)
        draw.line([(left, top), (left, bottom), (right, bottom),
                   (right, top), (left, top)], width=line_thickness, fill=color)
        draw_text(draw, box_to_display_str_map, box, left, right, top, bottom, color)


def main():
    # get devices
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    print("using {} device.".format(device))

    # load model
    # def load_checkpoint(filepath):
    #     checkpoint = torch.load(filepath)
    #     load_model = checkpoint['model']
    #     load_model.load_state_dict(checkpoint['state_dict'])
    #     for parameter in load_model.parameters():
    #         parameter.requires_grad = False
    #
    #     load_model.eval()
    #     return load_model
    #
    # model = load_checkpoint('checkpoint.pth')
    model_path = load_path("model.pkl")
    model = torch.load(model_path)
    model.to(device)
    # print(model)

    # read class_indict
    label_json_path = load_path("storefront_classes.json")
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    category_index = {v: k for k, v in class_dict.items()}

    # load image
    url1 = "https://firebasestorage.googleapis.com/v0/b/crsp--streetview.appspot.com/o/nH_LCOEEmGcR9l7PrJZv8w%2Fh_25.8786633181877_p_-3.1745163297469645?alt=media&token=cbd68e85-d111-4cc3-b306-d7ec98b67920"
    original_img = Image.open(load_path("test.jpg"))
    print(type(original_img))
    print(original_img)

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # print("pre image -> ", img)
    print("pre image shape -> ", img.shape)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    # print("after image -> ", img)
    print("after image shape -> ", img.shape)

    model.eval()  # evaluation
    with torch.no_grad():
        # init
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        predict_result = model(img.to(device))
        print("predict_result -> ", predict_result)

        predictions = model(img.to(device))[0]

        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        final_result = {"predict_boxes": predict_boxes, "predict_classes": predict_classes,
                        "predict_scores": predict_scores}
        print(final_result)
        if len(predict_boxes) == 0:
            print("no detections!")

        tensor_boxes = predictions["boxes"].to("cpu")
        tensor_scores = predictions["scores"].to("cpu")
        keep_index = torchvision.ops.nms(tensor_boxes, tensor_scores, 0.2 )
        print("NMS -> ", keep_index)


        draw_box(original_img,
                 predict_boxes[(keep_index.numpy().tolist())],
                 predict_classes[(keep_index.numpy().tolist())],
                 predict_scores[(keep_index.numpy().tolist())],
                 category_index,
                 thresh=0.2,
                 line_thickness=3)
        plt.imshow(original_img)
        plt.show()
        original_img.save("test_result.jpg")


if __name__ == '__main__':
    main()
