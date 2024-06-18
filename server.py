from flask import Flask, request
from keras.models import load_model
import numpy as np
from io import BytesIO
from flask_cors import CORS
import numpy as np
from PIL import Image


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

labels = ['Cam thảo', 'Cây bạc hà', 'Cây lưu ly', 'Cây mắc cỡ', 'Cây mã đề', 'Cỏ thơm', 'Đinh hương', 'Đông trùng hạ thảo', 'Hạt dẻ ngựa', 'Hoa đậu biếc', 'Ích mẫu', 'Ngải cứu', 'Nha đam', 'Tỏi', 'Trinh nữ hoàng cung', 'Ý dĩ']
model_saved = load_model('new_67_60.h5')

def handle_image(image_data):
  img = Image.open(image_data)
  img = img.convert('RGB')
  new_size = (224, 224)
  img_resized = img.resize(new_size)
  image_data = np.array(img_resized)
  image_data = np.expand_dims(image_data, axis=0)
  return image_data

@app.route('/predict-herb', methods=['POST'])
def prediction():
  image_file = request.files['image']
  image_data = BytesIO(image_file.read())
  
  image_data = handle_image(image_data)
  output = model_saved.predict(image_data)
  percent = output.max() * 100
  label_predict = labels[np.argmax(output)]

  return {"label": label_predict}

@app.route('/', methods=['GET'])
def Home():
    return "Ok"

if __name__ == '__main__':
    # app.run(host='0.0.0.0', port=443, ssl_context=('/app/server.crt', '/app/server.key'))
    app.run(debug=True, host='0.0.0.0', port=5000)