from PyPDF2 import PdfReader

import zipfile
import logging

logger = logging.getLogger(__name__)


def extract_text_from_pdf(file_instance):
    try:
        reader = PdfReader(file_instance.file.path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        return


def extract_text_from_zip(zip_instance):
    try:
        text_data = []
        with zipfile.ZipFile(zip_instance.file.path, 'r') as zip_ref:
            for file_name in zip_ref.namelist():
                if file_name.endswith('.pdf'):
                    with zip_ref.open(file_name) as pdf_file:
                        file_content = extract_text_from_pdf(pdf_file)
                        text_data.append(file_content)
        return "\n".join(text_data)
    except Exception as e:
        logger.error(f"Error extracting text from ZIP: {str(e)}")
        return
