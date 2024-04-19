import streamlit as st
import requests
import mimetypes
from PIL import Image

def main():

    st.image('https://www.iit.edu/themes/iit/assets/img/illinois-tech-red.svg',width = 300)
    st.title("Upload your files")
    st.write("File should be txt, docx, doc, pdf, zip")

    project_name = st.radio("Select project", ["gptzero", "BERT"], index=0)  # Default to gptzero

    uploaded_file = st.file_uploader("Upload a Word or PDF file", type=["docx", "pdf", "txt", "doc", "zip"])

    if uploaded_file is not None:
        # Create a FormData object containing the file and project name
        files = {'file': (uploaded_file.name, uploaded_file, get_content_type(uploaded_file))}  
        data = {'project': project_name}
        # Make an HTTP POST request to the Flask backend
        with st.spinner('Processing...'):
            # Make an HTTP POST request to the Flask backend
            # response = api_gptzero.extract_files(files)
            response = requests.post("http://localhost:5001/upload", files=files, data=data)

            if response.status_code == 200:
                # Download button for the processed file
                download_button = st.download_button(
                    label="Download Processed File",
                    data=response.content,
                    file_name='output.zip',
                    mime='application/zip'
                )
                if download_button:
                    st.write("Report Downloaded Successfully ✅")

            else:
                st.error("Error processing the file. Please try again later.")
                st.write("🦜 Something went wrong")

def get_content_type(uploaded_file):
    content_type, _ = mimetypes.guess_type(uploaded_file.name)
    return content_type or 'application/octet-stream'

if __name__ == "__main__":
    main()
