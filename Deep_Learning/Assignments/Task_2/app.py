import streamlit as st
from PIL import Image, ImageOps
import io
import numpy as np
import matplotlib.pyplot as plt
from typing import Any

# Try to import from tensorflow.keras, fall back to standalone keras if tensorflow is not available
try:
    from tensorflow.keras.datasets import mnist # type: ignore
    from tensorflow.keras.models import load_model # type: ignore
except Exception:
    from keras.datasets import mnist
    from keras.models import load_model

model: Any = None
st.title('Digit Recognizer — CNN')


col1, col2 = st.columns([1,1])
# keep a persistent uploaded_file variable
uploaded_file = None
with col1:
    uploaded_u = st.file_uploader('Upload a 28x28 grayscale or colored image of a digit (png/jpg)', type=['png','jpg','jpeg'])
    if st.button('Use example image'):
        # create a simple example from keras
        (x_train, y_train), _ = mnist.load_data()
        img = Image.fromarray((x_train[0]).astype('uint8'))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        uploaded_file = buf
    else:
        uploaded_file = uploaded_u


with col2:
    st.write('Instructions:')
    st.write('- Upload a photo (ideally centered on the digit).')
    st.write('- Or use the example image to test quickly.')


if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file).convert('L') # convert to grayscale
    st.image(image, caption='Input (grayscale)', width=150)


    # Preprocess: resize to 28x28, invert if needed, normalize
    img = ImageOps.invert(image) # MNIST digits are white on black — invert common photos
    # choose resampling filter in a backwards-compatible way: prefer Resampling.LANCZOS, fall back to Image.LANCZOS or BICUBIC
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:
        # avoid direct attribute access to Image.BICUBIC (some stubs/tools flag this); use getattr with a safe numeric default (3 == BICUBIC)
        resample = getattr(Image, "LANCZOS", getattr(Image, "BICUBIC", 3))
    img = img.resize((28,28), resample)
    arr = np.array(img).astype('float32') / 255.0
    arr = arr.reshape(1,28,28,1)


    if model is None:
        try:
            model = load_model(r'C:\Users\MAHADEV\Desktop\Tops\Deep_Learning\Assignments\Task_2\saved_models\mnist_cnn_model.keras')
        except Exception as e:
            st.error('Could not load model. Place mnist_cnn.keras in the same directory as app.py. Error: '+str(e))


    if model is not None:
        preds = model.predict(arr)
        pred_class = np.argmax(preds, axis=1)[0]
        st.subheader(f'Prediction: {pred_class}')


        # Display probabilities bar chart
        prob = preds[0]
        fig, ax = plt.subplots()
        ax.bar(range(10), prob)
        ax.set_xticks(range(10))
        ax.set_xlabel('Digit')
        ax.set_ylabel('Probability')
        st.pyplot(fig)


        st.write('Top 3 predictions:')
        top3_idx = prob.argsort()[-3:][::-1]
        for i in top3_idx:
            st.write(f'{i}: {prob[i]*100:.2f}%')


# Footer
st.markdown('---')
st.write('Tips: crop tightly, use plain background. If prediction is wrong, try inverting background or cleaning the input image.')