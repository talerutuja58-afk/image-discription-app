from flask import Flask, request, jsonify
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

app = Flask(__name__)

# Load AI model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.route("/upload", methods=["POST"])
def upload_image():
    file = request.files['image']
    file.save("uploaded.jpg")
    
    # Load image
    image = Image.open("uploaded.jpg").convert("RGB")
    
    # Generate caption
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)
    
    return jsonify({
        "description": caption
    })

if __name__ == "__main__":
    app.run(debug=True)
    