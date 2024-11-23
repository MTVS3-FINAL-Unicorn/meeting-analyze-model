import os
from collections import Counter

import numpy as np
from PIL import Image
from wordcloud import WordCloud


def make_wordcloud(tokens, mask_image_path=None, width=800, height=400, output_image_name='wordcloud.png'):
    if not os.path.isdir("./data"):
        os.mkdir("./data")
    output_image_path = f"./data/{output_image_name}"
    mask_image = np.array(Image.open(mask_image_path)) if mask_image_path else None
    token_counts = Counter(tokens)
    wordcloud = WordCloud(
        font_path='./static/fonts/NanumGothic.ttf', 
        width=width, 
        height=height, 
        background_color='white', 
        mask=mask_image
        ).generate_from_frequencies(token_counts)
    wordcloud.to_file(output_image_path)
    
    print(f"Wordcloud image saved to {output_image_path}")
