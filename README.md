# ask-question-image-web-app-streamlit-langchain


<p align="center">
<a href="https://www.youtube.com/watch?v=71EOM5__vkI">
    <img width="600" src="https://utils-computervisiondeveloper.s3.amazonaws.com/thumbnails/with_play_button/ask_question_image.jpg" alt="Watch the video">
    </br>Watch on YouTube: Ask questions to an image using Python, Streamlit and Langchain !
</a>
</p>

This is a Streamlit application that allows users to ask questions about an uploaded image and receive responses from a conversational AI agent. The agent uses the OpenAI GPT-3.5 Turbo model to generate answers based on the provided image and user input.

## installation

1. Clone the repository:

        git clone https://github.com/your-username/image-question-answering.git
        
2. Change to the project directory:

        cd  ask-question-image-web-app-streamlit-langchain
        
3. Install the required dependencies:

        pip install -r requirements.txt

4. Obtain an **OpenAI API key**. You can sign up for an API key at [OpenAI](https://platform.openai.com).

5. Replace the placeholder API key in the main.py file with your actual OpenAI API key:

        llm = ChatOpenAI(
            openai_api_key='YOUR_API_KEY',
            temperature=0,
            model_name="gpt-3.5-turbo"
        )

6. Run the Streamlit application:

        streamlit run main.py

7. Open your web browser and go to http://localhost:8501 to access the application.

## usage

1. Upload an image by clicking the file upload button.

2. The uploaded image will be displayed.

3. Enter a question about the image in the text input field.

4. The conversational AI agent will generate a response based on the provided question and image.

5. The response will be displayed below the question input.

## tools

The application utilizes the following custom tools:

- **ImageCaptionTool**: Generates a textual caption for the uploaded image.
- **ObjectDetectionTool**: Performs object detection on the uploaded image and identifies the objects present.

## contributing

Contributions are welcome! If you have any ideas, improvements, or bug fixes, please submit a pull request.

## license

This project is licensed under the MIT License.

## acknowledgements

This project uses the OpenAI GPT-3.5 Turbo model. Visit [OpenAI](https://openai.com/) for more information.

The Streamlit library is used for building the interactive user interface. Visit the [Streamlit documentation](https://docs.streamlit.io/) for more information.

The LangChain library is used for managing the conversational AI agent and tools. Visit the [LangChain GitHub repository](https://github.com/hwchase17/langchain) for more information.
    
The Transformers library is used to inference the AI features. Visit [this](https://huggingface.co/Salesforce/blip-image-captioning-large) and [this](https://huggingface.co/facebook/detr-resnet-50) pages for a more comprehensive description of the models used.
