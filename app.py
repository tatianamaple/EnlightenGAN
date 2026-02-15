from flask import Flask, request, render_template, send_file
import torch
from PIL import Image
import io
import torchvision.transforms as transforms
from models import networks
import torch.nn as nn

app = Flask(__name__)


# Initialize your generator
netG = networks.define_G(input_nc=3, output_nc=3, ngf=64, which_model_netG='sid_unet_resize', norm='instance', gpu_ids=[], opt=None)
#input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[], skip=False, opt=None


# Load your pretrained weights
state_dict = torch.load('checkpoints/enlightening/200_net_G_A.pth', map_location='cpu')
netG.load_state_dict(state_dict)
netG.eval()



def enhance_image(input_image):
    transform = transforms.ToTensor()
    inverse_transform = transforms.ToPILImage()

    input_tensor = transform(input_image).unsqueeze(0)  # Batch dimension
    with torch.no_grad():
        output_tensor = netG(input_tensor)
    output_image = inverse_transform(output_tensor.squeeze(0))
    return output_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        if file:
            image = Image.open(file.stream)
            enhanced_image = enhance_image(image)
            # Save to a bytes buffer
            buf = io.BytesIO()
            enhanced_image.save(buf, format='JPEG')
            buf.seek(0)
            return send_file(buf, mimetype='image/jpeg')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
