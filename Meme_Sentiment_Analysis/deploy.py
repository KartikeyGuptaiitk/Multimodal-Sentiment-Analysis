import streamlit as st
import torch
from PIL import Image, ImageFile
from transformers import AutoModel, DistilBertTokenizer, ViTImageProcessor
import io
import logging

# Enable truncated image loading
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Define model architecture with proper imports
class MultimodalModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.text_model = AutoModel.from_pretrained('distilbert-base-uncased')
        self.image_model = AutoModel.from_pretrained('google/vit-base-patch16-224-in21k')
        
        self.fusion = torch.nn.Sequential(
            torch.nn.Linear(768*2, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2)
        )
        
        self.classifier = torch.nn.ModuleDict({
            'sentiment': torch.nn.Linear(512, 5),
            'humor': torch.nn.Linear(512, 4),
            'sarcasm': torch.nn.Linear(512, 4),
            'offensive': torch.nn.Linear(512, 4),
            'motivational': torch.nn.Linear(512, 2)
        })

    def forward(self, input_ids, attention_mask, pixel_values):
        text_out = self.text_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:,0,:]
        image_out = self.image_model(pixel_values=pixel_values).last_hidden_state[:,0,:]
        
        fused = torch.cat([text_out, image_out], dim=1)
        fused = self.fusion(fused)
        
        return {task: self.classifier[task](fused) for task in self.classifier}

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalModel()
    model.load_state_dict(torch.load('memotion_model.pth', map_location=device))
    model.eval()
    return model.to(device)

model = load_model()
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
image_processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')

label_maps = {
    'sentiment': ['very_negative', 'negative', 'neutral', 'positive', 'very_positive'],
    'humor': ['not_funny', 'funny', 'very_funny', 'hilarious'],
    'sarcasm': ['not_sarcastic', 'general', 'twisted_meaning', 'very_twisted'],
    'offensive': ['not_offensive', 'slight', 'very_offensive', 'hateful_offensive'],
    'motivational': ['not_motivational', 'motivational']
}

st.title("Multimodal Meme Analysis")
st.write("Analyze memes for sentiment, humor, sarcasm, offensiveness, and motivation")

uploaded_file = st.file_uploader("Upload meme image", type=["jpg", "png", "jpeg"])
text_input = st.text_input("Enter meme text/caption", "")

if st.button("Analyze Meme"):
    if uploaded_file and text_input:
        try:
            device = next(model.parameters()).device
            
            # Process image
            img_bytes = uploaded_file.getvalue()
            with Image.open(io.BytesIO(img_bytes)) as img:
                image = img.convert('RGB')
                st.image(image, caption='Uploaded Meme', use_container_width=True)
                pixel_values = image_processor(images=image, return_tensors="pt", do_rescale=False).pixel_values.to(device)

            # Process text
            inputs = tokenizer(
                text_input,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            ).to(device)

            # Run inference
            with torch.no_grad():
                outputs = model(
                    input_ids=inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    pixel_values=pixel_values
                )

            # Display results
            st.subheader("Analysis Results")
            
            cols = st.columns(2)
            with cols[0]:
                st.metric("Sentiment", label_maps['sentiment'][outputs['sentiment'].argmax().item()])
                st.metric("Humor", label_maps['humor'][outputs['humor'].argmax().item()])
                
            with cols[1]:
                st.metric("Sarcasm", label_maps['sarcasm'][outputs['sarcasm'].argmax().item()])
                st.metric("Offensiveness", label_maps['offensive'][outputs['offensive'].argmax().item()])
                
            st.metric("Motivational", label_maps['motivational'][outputs['motivational'].argmax().item()])

        except Exception as e:
            logging.error(f"Error: {str(e)}")
            st.error(f"Analysis failed: {str(e)}")
    else:
        st.warning("Please upload an image and enter text for analysis")
