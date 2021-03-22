import sys
from pathlib import Path
import aiohttp
import asyncio
import uvicorn
from io import BytesIO
import base64


from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.templating import Jinja2Templates

import requests
import numpy as np
from PIL import Image
import torch
from icevision.core import ClassMap
from icevision.data import Dataset
from icevision.models import faster_rcnn
from icevision import tfms
from icevision.utils import denormalize_imagenet

# import requests
# import PIL, requests
# import icedata

templates = Jinja2Templates(directory='app/templates')

classes = ['black', 'grizzly', 'teddys']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))

def image_from_url(url):
    res = requests.get(url, stream=True)
    return Image.open(res.raw)

def image_to_base_64(url=None, image=None):
    if image is not None:
        the_image = image
    else:
        the_image = image_from_url(url)
    buffered = BytesIO()
    the_image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def calculate_padding(width, height, target_size):
    if width == height:
        return 0, 0
    
    if width > height:
        return 0, (target_size -  height / (width / target_size)) / 2
    else:
        return (target_size -  width / (height / target_size)) / 2, 0

def calculate_scaling(width, height, target_size, padding):
    padding_x, padding_y = padding
    return width / (target_size - 2 * padding_x), height / (target_size - 2 * padding_y)

def predict(model, urlOrImage):
    img = image_from_url(urlOrImage) if isinstance(urlOrImage, str) else urlOrImage
    width, height = img.size

    padding_x, padding_y = calculate_padding(width, height, 384)
    x_scale_ratio, y_scale_ratio = calculate_scaling(width, height, 384, (padding_x, padding_y))

    print('original')
    print(width)
    print(height)

    infer_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(384), tfms.A.Normalize()])
    infer_ds = Dataset.from_images([np.array(img)], infer_tfms)

    batch, samples = faster_rcnn.build_infer_batch(infer_ds)
    predictions = faster_rcnn.predict(model=model, batch=batch)
    boxes = []

    print(predictions)

    # birds only (label 3)
    for prediction in predictions:
        for label_index in range(len(prediction['labels'])):
            label = prediction['labels'][label_index]
            if label == 3:
                boxes.append(prediction['bboxes'][label_index])

    def scale_box(box):
        return {
            'xmin': (box.xmin - padding_x) * x_scale_ratio,
            'xmax': (box.xmax - padding_x) * x_scale_ratio,
            'ymin': (box.ymin - padding_y) * y_scale_ratio,
            'ymax': (box.ymax - padding_y) * y_scale_ratio
        }

    ## run transforms for all the boxes returned 
    ## look into the source of resize and pad to clean up implementation
    
    return predictions, Image.fromarray((denormalize_imagenet(samples[0]["img"])).astype(np.uint8)), list(map(scale_box, boxes) )

async def setup_model():
    _CLASSES = sorted(
        {
            "person",
            "bird",
            "cat",
            "cow",
            "dog",
            "horse",
            "sheep",
            "aeroplane",
            "bicycle",
            "boat",
            "bus",
            "car",
            "motorbike",
            "train",
            "bottle",
            "chair",
            "diningtable",
            "pottedplant",
            "sofa",
            "tvmonitor",
        }
    )

    data_class_map = ClassMap(classes=_CLASSES, background=0)
    model = faster_rcnn.model(num_classes=len(data_class_map))

    WEIGHTS_URL = "https://model-zoozoo.s3.amazonaws.com/pascal_faster_rcnn.pth"
    state_dict = torch.hub.load_state_dict_from_url(WEIGHTS_URL, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    return model

    # await download_file(export_file_url, path / export_file_name)
    # learn = load_learner(path, export_file_name)
    #return learn
    

loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_model())]
model = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())

@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = Image.open(BytesIO(img_bytes))
    predictions, tranformed_image, boxes = predict(model, img)
    return JSONResponse({'result': 'prediction', 'boxes': boxes})

@app.route('/icevision', methods=['GET'])
async def icevision(request):
    return templates.TemplateResponse('image.html', { 'request': request })
