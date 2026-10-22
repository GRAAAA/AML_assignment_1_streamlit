# Streamlit page for my CUB 200 bird classifier (COS30082 assignment 1)
# you upload a photo or pick an example, and it shows the top 5 species plus a Grad-CAM heatmap
from pathlib import Path

import streamlit as st
from PIL import Image

import bird_model

HERE = Path(__file__).parent
SETTINGS = bird_model.SETTINGS

st.set_page_config(page_title='Bird species classifier', page_icon='🐦', layout='wide')


@st.cache_resource(show_spinner='Loading the model, the first visit takes about a minute...')
def get_model():
    # cached, so the 352 MB model is only downloaded and loaded once per server and not on every click
    return bird_model.load_model()


st.title('Bird species classifier')
st.write(f"Upload a bird photo and my model guesses which of the 200 species in the CUB 200 dataset it is. "
         f"It's a {SETTINGS['backbone']} pretrained on ImageNet 22k and fine tuned on the bird photos "
         f"({SETTINGS['score'].replace('Top-1', 'Top 1')}).")

uploaded = st.file_uploader('Upload a bird photo', type=['jpg', 'jpeg', 'png'])
examples = sorted((HERE / 'examples').glob('*.jpg'))
choice = st.selectbox('or try one of the test photos (the model never trained on these)',
                      ['none'] + [p.stem for p in examples])

picture = None
if uploaded is not None:
    picture = Image.open(uploaded)
elif choice != 'none':
    picture = Image.open(HERE / 'examples' / f'{choice}.jpg')

if picture is not None:
    model = get_model()
    with st.spinner('Thinking...'):
        predictions = bird_model.top5(model, picture)
    best_name, best_p = predictions[0]

    photo_col, heat_col, guess_col = st.columns([1, 1, 1.2])
    photo_col.image(picture, caption='Your photo', width='stretch')
    with heat_col:
        with st.spinner('Making the heatmap...'):
            heat = bird_model.gradcam(model, picture)
        st.image(heat, caption=f'Where the model looked for "{best_name}" (the {bird_model.SIZE}px crop it saw)', width='stretch')

    with guess_col:
        st.subheader(best_name)
        st.write(f'{best_p:.1%} sure')
        if best_p < 0.5:
            # under 50% usually means two or three species that look almost the same are competing
            st.warning('Not very sure about this one. The other guesses below look similar, so check them too.')
        st.write('Top 5 guesses')
        for name, p in predictions:
            st.progress(p, text=f'{name}: {p:.1%}')

st.caption('COS30082 assignment 1, bird classification with transfer learning.')
