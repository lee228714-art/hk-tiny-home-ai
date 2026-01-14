import streamlit as st
from PIL import Image
import io
from huggingface_hub import InferenceClient

st.title("🛋️ GenAI 香港細單位室內設計助手")
st.write("Upload 單位相，選風格，用免費 Hugging Face AI 生成香港優化概念！（完全免費、無限試用，Stable Diffusion XL 模型）專為 300-500呎細單位。")

# Sidebar for Hugging Face Token (optional for public models, but add for safety)
if "HF_TOKEN" not in st.session_state:
    hf_token_input = st.sidebar.text_input("輸入 Hugging Face Token (optional, 免費註冊得)", type="password")
    if st.sidebar.button("Save Token"):
        st.session_state.HF_TOKEN = hf_token_input if hf_token_input else None
        st.rerun()
else:
    st.sidebar.success("Hugging Face Token 已設定（或用匿名）！")

# 選風格
style_options = ["Minimalist", "Japandi", "Biophilic (自然風)", "Industrial"]
selected_style = st.selectbox("選擇風格", style_options)

# Upload 圖片
uploaded_file = st.file_uploader("Upload 單位相或平面圖 (optional)", type=["jpg", "png", "jpeg"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Before: 原有空間", use_column_width=True)

# 輸入描述
description = st.text_input("加啲描述 (e.g., 多功能床、環保物料、防颱窗)", "香港細單位，優化空間，多功能傢俬")
room_desc = st.text_area("空間描述（英文更好，optional）", "a compact Hong Kong bedroom with bed, window, desk, backpack, narrow layout, natural light from small window")

# 組合 prompt
final_prompt = (
    f"Photorealistic redesign of this compact Hong Kong apartment (300-500 sqft): {room_desc} "
    f"Transform into beautiful {selected_style.lower()} style interior, {description}, "
    "smart multifunctional furniture (e.g. foldable bed, wall-mounted table), typhoon-resistant features (secure windows, sturdy fixtures), "
    "sustainable eco-friendly materials (bamboo, recycled wood), maximize natural light with mirrors and light colors, "
    "create spacious illusion in small space (clever storage, open layout), high-density urban Hong Kong vibe, "
    "typical HK flat elements like MTR proximity feel, Mong Kok energy, bright and airy, clean modern look, "
    "professional interior photography, ultra realistic, sharp focus, detailed textures, natural shadows, 8k resolution"
)

if st.button("🚀 生成 AI 概念圖（免費 SDXL）"):
    with st.spinner("AI 生成緊香港靚概念... 等陣啦～（幾秒到20秒）"):
        try:
            # 用 public stable model（無 gated）
            client = InferenceClient(token=st.session_state.get("HF_TOKEN"))
            image = client.text_to_image(
                model="stabilityai/stable-diffusion-xl-base-1.0",  # 穩定、免費、無 401
                prompt=final_prompt,
                height=1024,
                width=1024,
                num_inference_steps=30,  # 步驟多啲質素好
                guidance_scale=7.5
            )

            with col2:
                st.image(image, caption=f"After: {selected_style} 香港細單位概念", use_column_width=True)

            st.success("生成成功！如果圖唔夠靚，試加 prompt 細節如 'highly detailed, realistic lighting' 或等幾分鐘再試～")

        except Exception as e:
            st.error(f"哎呀！生成出錯：{str(e)}\n如果 401/unauthorized：試重新 generate Hugging Face token（去 https://huggingface.co/settings/tokens），或 token 留空用匿名（public model OK）。常見：quota 忙（等陣）、token 錯。")

else:
    st.info("填 prompt + 按掣，即刻生成香港靚概念！（完全免費）")

# Footer
st.markdown("---")
st.write("Prototype for Final Project – Powered by Streamlit & Hugging Face Stable Diffusion XL (完全免費開源). 未來可加 Flux 或其他模型！")