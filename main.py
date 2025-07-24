import json
from dotenv import load_dotenv
import os
from huggingface_hub import login 
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from transformers import BitsAndBytesConfig
from qwen_vl_utils import process_vision_info
import streamlit as st
from PIL import Image
import tempfile
import torch




load_dotenv('/home/lucifer/Documents/Invoice_Generation/.env')
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

login(token=HUGGINGFACE_TOKEN)

schema_dict = {
    "header": {
        "invoice_no": "Invoice number of the document",
        "invoice_date": "Date when the invoice was issued",
        "seller": "Full address and name of the seller",
        "client": "Full address and name of the client",
        "seller_tax_id": "Tax identification number of the seller",
        "client_tax_id": "Tax identification number of the client",
        "iban": "Bank IBAN number of the seller"
    },
    "items": [
        {
            "item_desc": "Description of the service or product sold",
            "item_qty": "Quantity of the item (usually in units or pieces)",
            "item_net_price": "Unit price excluding VAT",
            "item_net_worth": "Total price excluding VAT",
            "item_vat": "VAT rate applied to this item",
            "item_gross_worth": "Total price including VAT"
        }
    ],
    "summary": {
        "total_net_worth": "Total net amount before VAT",
        "total_vat": "Total VAT amount",
        "total_gross_worth": "Final total amount including VAT"
    }
}

instruction = f"""You are a specialized in invoice and your role is to extract information from any invoice that is provided to you in the following valid json format. if the corresponding value is not present, leave the key with empty string.

{json.dumps(schema_dict)}

Fill the keys only when the information is available.
"""


@st.cache_resource
def load_model():
    model_id = "Adarsh203/qwen-2.5-vl-3b-invoices"

    device_available = torch.cuda.is_available()

    if device_available:

    
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype="float16",
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            quantization_config=quant_config,
            device_map="auto")
    else:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="cpu")
        
    processor = AutoProcessor.from_pretrained(model_id, use_fast=True)

    return model,processor

model,processor = load_model()


def create_messages(image_path, prompt):
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_path,
                    "resized_height": 640,
                    "resized_width": 640
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return messages
    
def infer(messages, max_new_tokens=2048):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]



st.set_page_config(page_title="Invoice Extractor", layout="wide")
st.title("📄 Invoice Information Extraction")
st.markdown("Upload an invoice image to extract structured data in JSON format.")

if "parsed_output" not in st.session_state:
    st.session_state.parsed_output = None
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "reset_trigger" not in st.session_state:
    st.session_state.reset_trigger = False

uploaded_image = st.file_uploader("Upload Invoice Image", type=["png", "jpg", "jpeg"],
key=f"file_uploader_{st.session_state.reset_trigger}")
if uploaded_image:
    st.session_state.uploaded_image = Image.open(uploaded_image).convert("RGB")

if st.session_state.uploaded_image:
    st.markdown("---")
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🖼 Uploaded Invoice")
        st.image(st.session_state.uploaded_image, caption="Invoice Image", use_container_width=True)

    with col2:
        st.subheader("🧾 Extracted Data")
        if st.button("Extract Data"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                st.session_state.uploaded_image.save(tmp.name)
                image_path = tmp.name
            with st.spinner("Extracting data..."):
                messages = create_messages(image_path, instruction)
                st.session_state.parsed_output = infer(messages)
            st.rerun()

        if st.session_state.parsed_output:
            try:
                st.json(json.loads(st.session_state.parsed_output))
            except:
                st.text_area("Extracted Text", st.session_state.parsed_output, height=300)

    st.markdown("### 🔄 Reset")
    if st.button("Reset All"):
        st.session_state.parsed_output = None
        st.session_state.uploaded_image = None
        st.session_state.reset_trigger = not st.session_state.reset_trigger
        st.rerun()