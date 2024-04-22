import streamlit as st
import mimetypes
from PIL import Image
import api_gptzero
import io

def main():
    st.image("illinois-tech-with-seal.svg", width=300)
    # st.image('https://www.iit.edu/themes/iit/assets/img/illinois-tech-red.svg',width = 300)
    uploaded_file = st.file_uploader("Upload a Word, Text, Zip or PDF file", type=["docx", "pdf", "txt", "doc", "zip"])

    if uploaded_file is not None:
        with st.spinner('Processing...'):
            processed_file_data = api_gptzero.extract_files(uploaded_file)
            processed_file = io.BytesIO(processed_file_data)            

            if processed_file is not None:
                st.success("File Processed Successfully ✅")
                st.write("Download the below Output file")
                download_button = st.download_button(
                label="Download Processed File",
                data=processed_file,
                file_name='output.zip',
                mime='application/zip'
                )
                if download_button:
                    st.write("Report Downloaded Successfully ✅")

            else:
                st.error("Error processing the file. Please try again later.")
                st.write("🦜 Something went wrong")
    st.write("For more information, please visit [visit this site](https://sites.google.com/iit.edu/chatgpt-detector/home).")

def get_content_type(uploaded_file):
    content_type, _ = mimetypes.guess_type(uploaded_file.name)
    return content_type or 'application/octet-stream'

if __name__ == "__main__":
    main()
