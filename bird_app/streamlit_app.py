from pathlib import Path

import streamlit as st
from PIL import Image

import bird_model

st.set_page_config(page_title='Bird species classifier', page_icon='🐦')


@st.cache_resource(show_spinner='Loading the model (first visit takes about a minute)...')
def get_model():
    return bird_model.load_model()


st.title('Bird species classifier')
st.write(f"Recognises 200 bird species (CUB-200). Model: {bird_model.SETTINGS['backbone']} "
         f"fine-tuned with transfer learning, {bird_model.SETTINGS['score']}.")

uploaded = st.file_uploader('Upload a bird photo', type=['jpg', 'jpeg', 'png'])
examples = sorted((Path(__file__).parent / 'examples').glob('*.jpg'))
choice = st.selectbox('or try one of the test photos', ['none'] + [p.stem for p in examples])

picture = None
if uploaded is not None:
    picture = Image.open(uploaded)
elif choice != 'none':
    picture = Image.open(Path(__file__).parent / 'examples' / f'{choice}.jpg')

if picture is not None:
    left, right = st.columns(2)
    left.image(picture)
    with right:
        with st.spinner('Classifying...'):
            predictions = bird_model.top5(get_model(), picture)
        st.subheader(predictions[0][0])
        for name, p in predictions:
            st.write(f'{name}: {p:.1%}')
            st.progress(p)

st.caption('A clear photo of one bird works best. The model only knows these 200 species, '
           'so any other bird gets the closest match.')
